# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

from typing import Tuple
import torch
from torch import Tensor
import triton
import triton.language as tl
from triton.language.extra import libdevice
from .triton_kernels.utils import (
    IS_HIP, LazyDeviceTensors, get_num_SMs, next_power_of_2, get_closest_m,
)
from .dtypes import *

GEMLITE_ENABLE_PTX_FP4_PACK = False # Enable with CUDA13+ ptxas
def set_ptx_fp4_pack_flag(enabled: bool):
    global GEMLITE_ENABLE_PTX_FP4_PACK
    GEMLITE_ENABLE_PTX_FP4_PACK = enabled

#Get dtype min/max range based on compute dtype
def get_dtype_range(compute_dtype: torch.dtype) -> float:
    if(compute_dtype.is_floating_point):
        dtype_info = torch.finfo(compute_dtype)
    else:
        dtype_info = torch.iinfo(compute_dtype)
    return dtype_info.min, dtype_info.max

####################################################################################################################
#MXFP4 / NVFP4 weight quantizer
####################################################################################################################

#Cache workspace for multiple gpus (less than a KB per GPU).
#LazyDeviceTensors: allocated on FIRST USE, never at import , import-time
#cudaMalloc crashes under the GreenBoost LD_PRELOAD shim and breaks CPU-only
#imports. Call sites keep indexing with [device_index].
fp4_values = LazyDeviceTensors(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
    torch.float32,
)
fp4_p_vals = LazyDeviceTensors(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6],
    torch.float32,
)
fp4_thresholds = LazyDeviceTensors(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    torch.float32,
)  # (fp4_p_vals[:-1] + fp4_p_vals[1:]) / 2
thr_pos = LazyDeviceTensors(
    #last val is dummy to make len a power of 2
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0],
    torch.float32,
)  # (fp4_p_vals[:-1] + fp4_p_vals[1:]) / 2

class WeightQuantizerMXFP:
    def __init__(self, compute_dtype=torch.bfloat16, device="cuda:0"):
        self.compute_dtype = compute_dtype
        self.device        = device

    def round_to_closest_fp4(self, tensor):
        device_index = tensor.device.index
        out = fp4_p_vals[device_index][
            torch.searchsorted(
                fp4_thresholds[device_index].to(tensor.dtype), tensor.abs()
            )
        ].to(tensor.dtype)
        out *= tensor.sign()
        return out

    def to_index(self, W_q):
        assert W_q.is_floating_point(), "Input should be floating point fp4 values."
        device_index = W_q.device.index
        return (
            (W_q.view(-1, 1) == fp4_values[device_index].to(W_q.dtype).view(1, -1))
            .to(torch.uint8)
            .argmax(dim=1)
            .to(torch.uint8)
            .view(W_q.shape)
        )

    @torch.compile(fullgraph=True)
    def quantize_mxfp8(
        self,
        W: torch.Tensor,
        index: bool = False,
        mx_fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> (torch.Tensor, torch.Tensor):
        group_size: int = 32
        eps_exp: int = -30
        eps: float = 2 ** eps_exp
        min_val = torch.finfo(mx_fp8_dtype).min
        max_val = torch.finfo(mx_fp8_dtype).max

        W_flat = W.view(-1, group_size).float()
        ideal_scale = W_flat.abs().amax(dim=1, keepdim=True)
        ideal_scale /= max_val

        scales = (2 ** torch.ceil(torch.log2(ideal_scale))).clamp_(min=eps)

        W_q = (W_flat / scales).clamp_(min=min_val, max=max_val)
        scales = scales.to(torch.float8_e8m0fnu)

        if(index):
            W_q = W_q.to(mx_fp8_dtype)
        else:
            W_q = W_q.to(mx_fp8_dtype).to(W_flat.dtype)

        return W_q, scales
    
    @torch.compile(fullgraph=True)
    def quantize_mxfp4(
        self, W: torch.Tensor, window_size: int = 0, index: bool = False
    ) -> (torch.Tensor, torch.Tensor):
        group_size: int = 32
        eps_exp: int = -30
        eps: float = 2 ** eps_exp
        W_nbits = 4
        max_val = 6

        W_flat = W.view(-1, group_size).float()
        ideal_scale = W_flat.abs().amax(dim=1, keepdim=True)
        ideal_scale /= max_val

        if(window_size == 0):
            scales = 2 ** torch.ceil(torch.log2(ideal_scale))
        else:
            initial_log2_scales = torch.ceil(torch.log2(ideal_scale))
            search_offsets = torch.arange(
                -window_size,
                window_size + 1,
                device=W.device,
                dtype=initial_log2_scales.dtype,
            ).view(1, -1)
            candidate_scales = torch.pow(2, initial_log2_scales + search_offsets)
            candidate_scales[candidate_scales < eps] = eps

            W_q_candidates = self.round_to_closest_fp4(W_flat.unsqueeze(1) / candidate_scales.unsqueeze(-1))
            W_r_candidates = W_q_candidates * candidate_scales.unsqueeze(-1)
            errors = (W_flat.unsqueeze(1) - W_r_candidates).abs().mean(dim=-1)
            scales = torch.gather(candidate_scales, 1, torch.argmin(errors, dim=1, keepdim=True))

        scales = scales.clamp_(eps)
        W_q = self.round_to_closest_fp4(W_flat / scales)
        scales = scales.to(torch.float8_e8m0fnu)

        if(index):
            W_q = self.to_index(W_q)
        return W_q, scales
    
    # fullgraph=False: the fp8 reinterpret-views (Tensor.view(float8_e4m3fn))
    # are Unsupported under fullgraph dynamo on torch 2.11 ("torch.* op
    # returned non-Tensor").  Quantization is a one-time load step - graph
    # breaks are fine here.
    @torch.compile(fullgraph=False)
    def quantize_nvfp4(
        self, W: torch.Tensor, window_size: int = 0, index: bool = False,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):

        group_size: int = 16
        eps: float = 1e-6
        W_nbits = 4
        max_val = 6
        fp8_dtype = torch.float8_e4m3fn #This is for Nvidia only.
        max_fp8 = torch.finfo(fp8_dtype).max #448

        W_flat = W.view(-1, group_size).float()
        ideal_scale = W_flat.abs().amax(dim=1, keepdim=True)
        ideal_scale /= max_val
        meta_scales = (max_fp8 / ideal_scale.max().clamp_(min=eps)).float()
        ideal_scale *= meta_scales
        ideal_scale = ideal_scale.clamp_(max=max_fp8).to(fp8_dtype)

        if(window_size == 0):
            scales = ideal_scale
        else:
            search_offsets = torch.arange(
                -window_size, window_size + 1, device=W.device, dtype=torch.int
            ).view(1, -1)

            candidate_scales = (
                (ideal_scale.view(torch.int8) + search_offsets)
                .clamp_(-128, 127)
                .to(torch.int8)
            )

            #Avoid nan in int8 range (-1, 127 as int8 as e4m3 nans)
            candidate_scales[candidate_scales==-1] = 1
            candidate_scales[candidate_scales==127] = 1
            candidate_scales = candidate_scales.view(fp8_dtype).float()
            candidate_scales[candidate_scales < eps] = eps

            W_q_candidates = self.round_to_closest_fp4(W_flat.unsqueeze(1) * (meta_scales / candidate_scales).unsqueeze(-1))
            W_r_candidates = W_q_candidates * (candidate_scales / meta_scales).unsqueeze(-1)
            errors = (W_flat.unsqueeze(1) - W_r_candidates).abs().mean(dim=-1)
            scales = torch.gather(candidate_scales, 1, torch.argmin(errors, dim=1, keepdim=True)).to(fp8_dtype)

        scales_full = (scales.to(W_flat.dtype) / meta_scales).clamp_(min=eps)
        W_q = self.round_to_closest_fp4(W_flat / scales_full)

        if(index):
            W_q = self.to_index(W_q)

        return W_q, scales, meta_scales

    def dequantize(self, W_q, scales, shape = None, dtype = None, meta_scales = None):
        if(W_q.dtype == torch.uint8): #from indices
            device_index = W_q.device.index
            W_q = fp4_values[device_index][W_q.int()]

        group_size = W_q.numel() // scales.numel()
        out = (W_q.view([-1, group_size]).float() * scales.float())
        if meta_scales is not None:
            out = out * meta_scales
        if(shape is not None):
            out = out.view(shape)
        return out.to(self.compute_dtype if dtype is None else dtype)

####################################################################################################################
#INT8 / FP8 activations
####################################################################################################################
def prune_large_blocks(configs, named_args, **kwargs):
    M = named_args.get('M_CLOSEST', named_args.get('M'))
    
    pruned = []
    for config in configs:
        if config.kwargs['BLOCK_SIZE_M'] <= M:
            pruned.append(config)
            
    if not pruned:
        for config in configs:
            new_kwargs = config.kwargs.copy()
            new_kwargs['BLOCK_SIZE_M'] = 16
                
            pruned.append(
                triton.Config(
                    new_kwargs,
                    num_warps=config.num_warps,
                    num_stages=config.num_stages
                )
            )
        
    return pruned

# Main activation scaling functions
@torch.compile(fullgraph=True)
def scale_activations_per_token_torch(
    tensor: Tensor, w_dtype: torch.dtype, fp32_scale: bool = True
) -> Tuple[Tensor, Tensor]:

    min_val, max_val = get_dtype_range(w_dtype)
    if fp32_scale:
        tensor = tensor.to(torch.float32, copy=False)
    out_shape = tensor.shape
    out = tensor.view(-1, tensor.shape[-1])
    scales = torch.abs(out).amax(axis=1, keepdim=True)
    # if(fp32_scale):
    #     scales = scales.to(torch.float32)
    scales.div_(max_val)
    scales.clamp_(min=1e-6)
    out = tensor / scales
    out.clamp_(min_val, max_val)

    if not w_dtype.is_floating_point:
        out.round_()

    out = out.to(dtype=w_dtype)    
    return out.view(out_shape), scales

@triton.jit
def round_triton_nvidia(tensor):
    return libdevice.round(tensor)

@triton.jit
def round_triton_amd(tensor):
    return libdevice.floor(tensor + 0.50)

if IS_HIP:
    round_triton = round_triton_amd
else:
    round_triton = round_triton_nvidia

@triton.jit
def scale_activations_per_token_triton_v1_kernel(
    tensor_ptr, scale_ptr, y_ptr, 
    M, K,
    stride_m: tl.constexpr, 
    stride_k: tl.constexpr, 
    stride_sm: tl.constexpr,
    ROUND: tl.constexpr, 
    UNROLL: tl.constexpr,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    fp32_scale: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0) * UNROLL
    pid_k = tl.program_id(1)

    offs_k  = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_m  = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    for m in range(UNROLL):
        mask = ((offs_m < M)[:, None] & (offs_k < K)[None, :]).to(tl.int1)
        in_ptrs = offs_m[:, None] * stride_m + offs_k[None, :] * stride_k
        tensor = tl.load(tensor_ptr + in_ptrs, mask=mask, other=0.0)
        if fp32_scale:
            tensor = tensor.to(tl.float32)

        scales_x = tl.max(tl.abs(tensor), axis=1, keep_dims=True)
        scales_x /= max_val
        scales_x = tl.maximum(scales_x, 1e-6)
        tensor /= scales_x
        tensor = tl.minimum(tl.maximum(tensor, min_val), max_val)

        if ROUND:
            tensor = round_triton(tensor)

        tl.store(scale_ptr + offs_m[:, None] * stride_sm, scales_x)
        tl.store(y_ptr + in_ptrs, tensor, mask=mask)
        offs_m += BLOCK_SIZE_M

def scale_activations_per_token_triton_v1(
    tensor: Tensor, w_dtype: torch.dtype, fp32_scale: bool = True
) -> Tuple[Tensor, Tensor]:
    min_val, max_val = get_dtype_range(w_dtype)
    x_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape
    scales = torch.empty(
        (M, 1), dtype=torch.float32 if fp32_scale else tensor.dtype, device=tensor.device
    )
    y = torch.empty((M, K), dtype=w_dtype, device=tensor.device)

    UNROLL = 1  # max(1, M // 128)
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_K = triton.next_power_of_2(K)
    grid = (triton.cdiv(M, BLOCK_SIZE_M * UNROLL), triton.cdiv(K, BLOCK_SIZE_K))

    ROUND = not w_dtype.is_floating_point

    scale_activations_per_token_triton_v1_kernel[grid](
        tensor,
        scales,
        y,
        M,
        K,
        tensor.stride(0),
        tensor.stride(1),
        scales.stride(0),
        min_val=min_val,
        max_val=max_val,
        fp32_scale=fp32_scale,
        ROUND=ROUND,
        UNROLL=UNROLL,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_stages=1,
        num_warps=4,
    )

    return y.view(x_shape), scales

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 2}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 4}, num_warps=8, num_stages=1),
    ],
    key=['M_CLOSEST', 'K']
)
@triton.jit
def scale_activations_per_token_triton_v2_kernel(
    tensor_ptr, scale_ptr, y_ptr,
    M, K, M_CLOSEST,
    stride_m: tl.constexpr, 
    stride_k: tl.constexpr, 
    stride_sm: tl.constexpr,
    min_val: tl.constexpr, 
    max_val: tl.constexpr,
    fp32_scale: tl.constexpr, 
    ROUND: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)   
    m_mask = offs_m < M
    k_mask = offs_k < K
    mask = m_mask[:, None] & k_mask[None, :]

    offsets = offs_m[:, None] * stride_m + offs_k[None, :] * stride_k

    tensor = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
    
    if fp32_scale:
        tensor = tensor.to(tl.float32)

    scales_x = tl.max(tl.abs(tensor), axis=1) / max_val
    scales_x = tl.maximum(scales_x, 1e-6)
    tensor = tensor / scales_x[:, None]
    tensor = tl.minimum(tl.maximum(tensor, min_val), max_val)

    if ROUND:
        tensor = round_triton(tensor)

    tl.store(scale_ptr + offs_m * stride_sm, scales_x, mask=m_mask)
    tl.store(y_ptr + offsets, tensor, mask=mask)

def scale_activations_per_token_triton_v2(
    tensor: torch.Tensor, w_dtype: torch.dtype, fp32_scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    min_val, max_val = get_dtype_range(w_dtype)
    
    x_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape
    
    scales = torch.empty((M, 1), dtype=torch.float32 if fp32_scale else tensor.dtype, device=tensor.device)
    y = torch.empty((M, K), dtype=w_dtype, device=tensor.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), )

    BLOCK_SIZE_K = triton.next_power_of_2(K)
    ROUND = not w_dtype.is_floating_point

    M_CLOSEST = get_closest_m(M)
    scale_activations_per_token_triton_v2_kernel[grid](
        tensor, scales, y,
        M, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0),
        min_val=min_val, max_val=max_val,
        fp32_scale=fp32_scale, ROUND=ROUND,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return y.view(x_shape), scales

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 2}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 4}, num_warps=8, num_stages=1),
    ],
    key=['M_CLOSEST', 'K']
)
@triton.jit
def scale_activations_per_token_triton_v3_kernel(
    tensor_ptr, scale_ptr, y_ptr,
    M, K, M_CLOSEST,
    stride_m: tl.constexpr, 
    stride_k: tl.constexpr, 
    stride_sm: tl.constexpr,
    min_val: tl.constexpr, 
    max_val: tl.constexpr,
    fp32_scale: tl.constexpr, 
    ROUND: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    k_mask = offs_k < K

    for pid_m in range(start_pid, num_tiles, num_programs):
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = offs_m < M
        mask = m_mask[:, None] & k_mask[None, :]

        offsets = offs_m[:, None] * stride_m + offs_k[None, :] * stride_k
        tensor = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
        
        if fp32_scale:
            tensor = tensor.to(tl.float32)
            
        scales_x = tl.max(tl.abs(tensor), axis=1) / max_val
        scales_x = tl.maximum(scales_x, 1e-6)
        tensor = tensor / scales_x[:, None]
        tensor = tl.minimum(tl.maximum(tensor, min_val), max_val)

        if ROUND:
            tensor = round_triton(tensor)
        
        tl.store(y_ptr + offsets, tensor, mask=mask)
        tl.store(scale_ptr + offs_m * stride_sm, scales_x, mask=m_mask)
        
def scale_activations_per_token_triton_v3(
    tensor: torch.Tensor, w_dtype: torch.dtype, fp32_scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    min_val, max_val = get_dtype_range(w_dtype)
    
    x_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape
    
    scales = torch.empty((M, 1), dtype=torch.float32 if fp32_scale else tensor.dtype, device=tensor.device)
    y = torch.empty((M, K), dtype=w_dtype, device=tensor.device)

    grid = lambda META: (min(get_num_SMs(), triton.cdiv(M, META['BLOCK_SIZE_M'])), )

    BLOCK_SIZE_K = triton.next_power_of_2(K)
    ROUND = not w_dtype.is_floating_point

    M_CLOSEST = get_closest_m(M)
    scale_activations_per_token_triton_v3_kernel[grid](
        tensor, scales, y,
        M, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0),
        min_val=min_val, max_val=max_val,
        fp32_scale=fp32_scale, ROUND=ROUND,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return y.view(x_shape), scales


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1, "BLOCK_SIZE_K": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 1, "BLOCK_SIZE_K": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 2, "BLOCK_SIZE_K": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 2, "BLOCK_SIZE_K": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 4, "BLOCK_SIZE_K": 4096}, num_warps=8, num_stages=2),
    ],
    key=["M_CLOSEST", "K"],
)
@triton.jit
def scale_activations_per_token_triton_v4_kernel(
    tensor_ptr,
    scale_ptr,
    y_ptr,
    M,
    K,
    M_CLOSEST,
    stride_m: tl.constexpr,
    stride_k: tl.constexpr,
    stride_sm: tl.constexpr,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    fp32_scale: tl.constexpr,
    ROUND: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M)

    for pid_m in range(start_pid, num_tiles, num_programs):
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = offs_m < M

        # Pass 1: streaming amax over K chunks
        row_max = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_SIZE_K):
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = offs_k < K
            mask = m_mask[:, None] & k_mask[None, :]
            chunk = tl.load(
                tensor_ptr + offs_m[:, None] * stride_m + offs_k[None, :] * stride_k,
                mask=mask,
                other=0.0,
            )
            if fp32_scale:
                chunk = chunk.to(tl.float32)
            row_max = tl.maximum(row_max, tl.max(tl.abs(chunk), axis=1))

        scales_x = row_max / max_val
        scales_x = tl.maximum(scales_x, 1e-6)
        tl.store(scale_ptr + offs_m * stride_sm, scales_x, mask=m_mask)

        # Pass 2: scale, clamp, store
        inv_scales = 1.0 / scales_x
        for k_start in range(0, K, BLOCK_SIZE_K):
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = offs_k < K
            mask = m_mask[:, None] & k_mask[None, :]
            offsets = offs_m[:, None] * stride_m + offs_k[None, :] * stride_k
            chunk = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
            if fp32_scale:
                chunk = chunk.to(tl.float32)
            chunk = chunk * inv_scales[:, None]
            chunk = tl.minimum(tl.maximum(chunk, min_val), max_val)
            if ROUND:
                chunk = round_triton(chunk)
            tl.store(y_ptr + offsets, chunk, mask=mask)


def scale_activations_per_token_triton_v4(
    tensor: torch.Tensor,
    w_dtype: torch.dtype,
    fp32_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_val, max_val = get_dtype_range(w_dtype)

    x_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    scales = torch.empty(
        (M, 1),
        dtype=torch.float32 if fp32_scale else tensor.dtype,
        device=tensor.device,
    )
    y = torch.empty((M, K), dtype=w_dtype, device=tensor.device)

    grid = lambda META: (min(get_num_SMs(), triton.cdiv(M, META["BLOCK_SIZE_M"])),)

    ROUND = not w_dtype.is_floating_point

    M_CLOSEST = get_closest_m(M)
    scale_activations_per_token_triton_v4_kernel[grid](
        tensor,
        scales,
        y,
        M,
        K,
        M_CLOSEST,
        tensor.stride(0),
        tensor.stride(1),
        scales.stride(0),
        min_val=min_val,
        max_val=max_val,
        fp32_scale=fp32_scale,
        ROUND=ROUND,
    )

    return y.view(x_shape), scales
####################################################################################################################
#MXFP8
####################################################################################################################
@triton.jit
def next_power_of_2_log_triton(val, eps_exp: tl.constexpr):
    exp = tl.ceil(tl.log2(val)).to(tl.int32)
    exp = exp + 127
    exp = tl.maximum(tl.minimum(exp, 254), 127 + eps_exp)
    scales = tl.cast(exp << 23, tl.float32, bitcast=True)
    return scales, exp

@triton.jit
def next_power_of_2_ptx_triton(val, eps_exp: tl.constexpr):
    scales, biased_exp = tl.inline_asm_elementwise(
        f"""
        {{
        .reg .f32 f_log;
        .reg .f32 f_ceil;
        .reg .s32 r_exp;
        .reg .f32 f_clamped;

        lg2.approx.f32 f_log, $2;
        cvt.rpi.f32.f32 f_ceil, f_log;
        cvt.rzi.s32.f32 r_exp, f_ceil;

        max.s32 r_exp, r_exp, {eps_exp};
        min.s32 r_exp, r_exp, 127;

        add.s32 $1, r_exp, 127;
        cvt.rn.f32.s32 f_clamped, r_exp;
        ex2.approx.f32 $0, f_clamped;
        }}
        """,
        "=f,=r,f",
        [val],
        dtype=(tl.float32, tl.int32),
        is_pure=True,
        pack=1
    )
    
    return scales, biased_exp

@triton.jit
def next_power_of_2_bitwise_triton(val, eps_exp: tl.constexpr):
    xi = tl.cast(val, tl.uint32, bitcast=True)
    exp  = (xi >> 23) & 0xFF
    mant = xi & 0x7FFFFF
    exp += tl.where(mant != 0, 1, 0)
    exp = tl.maximum(tl.minimum(exp, 254), 127 + eps_exp)
    scales = tl.cast(exp << 23, tl.float32, bitcast=True)
    return scales, exp

next_power_of_2_triton = next_power_of_2_bitwise_triton

####################################################################################################################
#MXFP8
####################################################################################################################
@torch.compile(fullgraph=True)
def scale_activations_mxfp8_torch(
    tensor: Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[Tensor, Tensor]:

    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp
    min_val, max_val = get_dtype_range(w_dtype)

    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape

    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    if(pad_rows > 0):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    post_pad_shape = tensor.shape

    W_flat = tensor.view(-1, group_size).float()
    scales = W_flat.abs().amax(dim=1, keepdim=True)
    scales /= max_val
    scales = (2 ** torch.ceil(torch.log2(scales))).clamp_(eps) 

    W_q = (W_flat / scales).clamp_(min_val, max_val).to(w_dtype)
    if(pad_rows > 0):
        W_q = W_q.view(post_pad_shape)[:inter_shape[0], :]

    W_q = W_q.view(orig_shape)
    scales = (
        scales.to(torch.float8_e8m0fnu)
        .view(torch.uint8)
        .view(post_pad_shape[0], post_pad_shape[1] // group_size)
    )

    return W_q, scales

@triton.jit
def scale_activations_mxfp8_triton_v1_kernel(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    E,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    eps_exp: tl.constexpr,
    UNROLL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0) * UNROLL

    for m in range(UNROLL):
        offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = (offs < E).to(tl.int1)
        tensor = tl.load(tensor_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor)) / max_val, eps_exp)

        out = tensor / scales
        out = tl.clamp(out, min=min_val, max=max_val)
        out = out.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, out)
        tl.store(scales_ptr + pid, scales_log2)

        pid += 1

def scale_activations_mxfp8_triton_v1(
    tensor: torch.Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp
    min_val, max_val = get_dtype_range(w_dtype)
    tensor = tensor.contiguous()
    
    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape
    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    post_pad_shape = (inter_shape[0] + pad_rows, inter_shape[1])
    E = tensor.numel()

    UNROLL = min(triton.cdiv(triton.cdiv(E, group_size), get_num_SMs(tensor.device)), 1)

    out = torch.empty(inter_shape, device=tensor.device, dtype=w_dtype)

    scales = torch.empty(
        (post_pad_shape[0], post_pad_shape[1] // group_size),
        device=tensor.device,
        dtype=torch.uint8,
    )
    
    grid = lambda meta: (triton.cdiv(E // UNROLL, group_size), )
    scale_activations_mxfp8_triton_v1_kernel[grid](
                tensor, 
                out, 
                scales, 
                E=E,
                min_val=min_val,
                max_val=max_val,
                eps_exp=eps_exp,
                UNROLL=UNROLL,
                GROUP_SIZE=group_size,
                num_stages=1,
                num_warps=4,
                )

    return out.view(orig_shape), scales

@triton.jit
def scale_activations_mxfp8_triton_kernel_v2(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    M, K,
    #########################
    stride_m_t: tl.constexpr, 
    stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, 
    stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, 
    stride_k_o: tl.constexpr,
    #########################
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    #next power of 2 via log
    scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor), axis=1, keep_dims=True) / max_val, eps_exp)

    #Map to index
    out = tensor / scales
    out = tl.clamp(out, min=min_val, max=max_val)
    out = out.to(out_dtype)

    #Store
    out_mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k[None, :] * stride_k_o), out, mask=out_mask)

    offs_k = pid_k * 1 + tl.arange(0, 1)
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales_log2)


def scale_activations_mxfp8_triton_v2(
    tensor: torch.Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp
    min_val, max_val = get_dtype_range(w_dtype)

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K), device=tensor.device, dtype=w_dtype)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)
    
    #BLOCK_SIZE_M = min(max(next_power_of_2(M), group_size), 128)
    BLOCK_SIZE_M = group_size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, group_size))
    device_index = tensor.device.index

    scale_activations_mxfp8_triton_kernel_v2[grid](
        tensor,
        out,
        scales,
        M, K, 
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        #########################
        min_val=min_val,
        max_val=max_val,
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        num_stages=1,
        num_warps=4,
    )

    return out, scales

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4, num_stages=3),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_mxfp8_triton_kernel_v3(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    M, K, M_CLOSEST,
    #########################
    stride_m_t: tl.constexpr, 
    stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, 
    stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, 
    stride_k_o: tl.constexpr,
    #########################
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    tensor_block_ptr = tl.make_block_ptr(
        base=tensor_ptr, shape=(M, K), strides=(stride_m_t, stride_k_t),
        offsets=(pid_m * BLOCK_SIZE_M, pid_k * GROUP_SIZE),
        block_shape=(BLOCK_SIZE_M, GROUP_SIZE), order=(1, 0)
    )
    
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(M, K), strides=(stride_m_o, stride_k_o),
        offsets=(pid_m * BLOCK_SIZE_M, pid_k * GROUP_SIZE),
        block_shape=(BLOCK_SIZE_M, GROUP_SIZE), order=(1, 0)
    )

    tensor = tl.load(tensor_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    
    abs_max = tl.max(tl.abs(tensor), axis=1, keep_dims=True)
    scales, scales_log2 = next_power_of_2_triton(abs_max / max_val, eps_exp)

    out = tensor * (1.0 / scales)    
    out = tl.clamp(out, min=min_val, max=max_val)
    out = out.to(out_ptr.dtype.element_ty)

    tl.store(out_block_ptr, out, boundary_check=(0, 1))

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    scales_ptrs = scales_ptr + (offs_m * stride_m_s + pid_k * stride_k_s)    
    tl.store(scales_ptrs, tl.reshape(scales_log2, (BLOCK_SIZE_M, )), mask=mask_m)

def scale_activations_mxfp8_triton_v3(
    tensor: torch.Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp
    min_val, max_val = get_dtype_range(w_dtype)

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K), device=tensor.device, dtype=w_dtype)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(K, group_size))
    device_index = tensor.device.index

    M_CLOSEST = get_closest_m(M)
    scale_activations_mxfp8_triton_kernel_v3[grid](
        tensor,
        out,
        scales,
        M, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        #########################
        min_val=min_val,
        max_val=max_val,
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
    )

    return out, scales

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_K': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 8,  'BLOCK_SIZE_K': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 256}, num_warps=8, num_stages=1),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_mxfp8_triton_kernel_v4(
    tensor_ptr, out_ptr, scales_ptr,
    M, M_padded, K, M_CLOSEST,
    stride_m_t: tl.constexpr, stride_k_t: tl.constexpr,
    stride_m_o: tl.constexpr, stride_k_o: tl.constexpr,
    stride_m_s: tl.constexpr, stride_k_s: tl.constexpr,
    min_val: tl.constexpr, max_val: tl.constexpr,
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_m_tiles = tl.cdiv(M_padded, BLOCK_SIZE_M)

    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
    FLAT_M: tl.constexpr = BLOCK_SIZE_M * GROUPS_PER_BLOCK
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    for tile_m in range(pid, num_m_tiles, num_programs):
        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = offs_m < M

        tensor_bp = tl.make_block_ptr(
            tensor_ptr, (M, K), (stride_m_t, stride_k_t),
            (tile_m * BLOCK_SIZE_M, 0),
            (BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        out_bp = tl.make_block_ptr(
            out_ptr, (M, K), (stride_m_o, stride_k_o),
            (tile_m * BLOCK_SIZE_M, 0),
            (BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )

        for k_start in range(0, K, BLOCK_SIZE_K):
            # Load [BLOCK_M, BLOCK_K]
            tensor = tl.load(tensor_bp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

            # Reshape to [BLOCK_M * GROUPS_PER_BLOCK, GROUP_SIZE] for group-wise reduction
            tensor_flat = tl.reshape(tensor, (FLAT_M, GROUP_SIZE))

            # Per-group abs_max → power-of-2 scale
            abs_max = tl.max(tl.abs(tensor_flat), axis=1)
            scales, scales_log2 = next_power_of_2_bitwise_triton(abs_max / max_val, eps_exp)

            # Quantize: multiply by reciprocal, clamp, cast
            out = tensor_flat * (1.0 / scales[:, None])
            out = tl.clamp(out, min=min_val, max=max_val)
            out = out.to(out_dtype)

            # Reshape back to [BLOCK_M, BLOCK_K] and store
            out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_K))
            tl.store(out_bp, out, boundary_check=(0, 1))

            # Store scales: [FLAT_M] → [BLOCK_M, GROUPS_PER_BLOCK]
            scales_2d = tl.reshape(scales_log2, (BLOCK_SIZE_M, GROUPS_PER_BLOCK))
            # For padding rows (M <= row < M_padded), store identity scale (127 = 2^0 in E8M0)
            scales_2d = tl.where(m_mask[:, None], scales_2d, tl.full(scales_2d.shape, 127, dtype=tl.uint8))
            group_idx = k_start // GROUP_SIZE
            offs_g = group_idx + tl.arange(0, GROUPS_PER_BLOCK)
            g_mask = offs_g < tl.cdiv(K, GROUP_SIZE)
            tl.store(
                scales_ptr + offs_m[:, None] * stride_m_s + offs_g[None, :] * stride_k_s,
                scales_2d, mask=(offs_m[:, None] < M_padded) & g_mask[None, :]
            )

            tensor_bp = tl.advance(tensor_bp, (0, BLOCK_SIZE_K))
            out_bp = tl.advance(out_bp, (0, BLOCK_SIZE_K))

# ersistent 1D grid, processes multiple K-groups per iteration via reshape
def scale_activations_mxfp8_triton_v4(
    tensor: torch.Tensor, w_dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    min_val, max_val = get_dtype_range(w_dtype)

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K), device=tensor.device, dtype=w_dtype)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)

    grid = lambda meta: (min(get_num_SMs(), triton.cdiv(M, meta['BLOCK_SIZE_M'])),)

    M_CLOSEST = get_closest_m(M)
    scale_activations_mxfp8_triton_kernel_v4[grid](
        tensor, out, scales,
        M, M_padded, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        min_val=min_val, max_val=max_val,
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
    )

    return out, scales
####################################################################################################################
#MXPF4 / NVFP4
####################################################################################################################
@torch.compile(fullgraph=True)
def scale_activations_mxfp4_torch(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp
    max_val: float = 6

    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape

    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    if(pad_rows > 0):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    post_pad_shape = tensor.shape

    W_flat = tensor.view(-1, group_size).float()
    scales = W_flat.abs().amax(dim=1, keepdim=True)
    scales /= max_val
    scales = (2 ** torch.ceil(torch.log2(scales))).clamp_(eps)

    W_q = W_flat / scales
    if(pad_rows > 0):
        W_q = W_q.view(post_pad_shape)[:inter_shape[0], :]

    #1) Map to closest index
    device_index = W_q.device.index

    W_q = (
        (W_q.view(-1, 1) - fp4_values[device_index].to(W_q.dtype).view(1, -1))
        .abs()
        .argmin(dim=1)
        .to(torch.uint8)
        .view(inter_shape)
    )
    #2) Pack
    W_q = (W_q[:,::2] | W_q[:,1::2] << 4).to(torch.uint8)

    #Reshape scales
    scales = (
        scales.to(torch.float8_e8m0fnu)
        .view(torch.uint8)
        .view(post_pad_shape[0], post_pad_shape[1] // group_size)
    )
    return W_q, scales

@torch.compile(fullgraph=True)
def scale_activations_nvfp4_torch(tensor: Tensor, meta_scale=None) -> Tuple[Tensor, Tensor]:
    group_size: int = 16
    eps: float = 1e-6
    max_val: float = 6
    fp8_dtype = torch.float8_e4m3fn #Support Nvidia only
    max_fp8 = torch.finfo(fp8_dtype).max #448

    orig_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    inter_shape = tensor.shape

    pad_rows = (group_size - inter_shape[0] % group_size) % group_size
    if(pad_rows > 0):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))
    post_pad_shape = tensor.shape

    W_flat = tensor.view(-1, group_size).float()
    scales = W_flat.abs().amax(dim=1, keepdim=True)
    scales /= max_val
    meta_scales = meta_scale if meta_scale is not None else (max_fp8 / scales.max().clamp_(min=eps)).float()
    scales *= meta_scales
    scales = scales.clamp(max=max_fp8).to(fp8_dtype).to(W_flat.dtype)

    W_q = W_flat / (scales / meta_scales)
    if(pad_rows > 0):
        W_q = W_q.view(post_pad_shape)[:inter_shape[0], :]

    #1) Map to closest index
    device_index = W_q.device.index

    W_q = (
        (W_q.view(-1, 1) - fp4_values[device_index].to(W_q.dtype).view(1, -1))
        .abs()
        .argmin(dim=1)
        .to(torch.uint8)
        .view(inter_shape)
    )
    #2) Pack
    W_q = (W_q[:,::2] | W_q[:,1::2] << 4).to(torch.uint8)

    #Reshape scales
    scales = (
        scales
        .to(fp8_dtype)
        .view(post_pad_shape[0], post_pad_shape[1] // group_size)
    )
    return W_q, scales, meta_scales.float()

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4, num_stages=3),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_mxfp4_triton_kernel(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    thr_pos_ptr,
    M, K, M_CLOSEST,
    #########################
    stride_m_t: tl.constexpr, 
    stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, 
    stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, 
    stride_k_o: tl.constexpr,
    #########################
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    use_tma: tl.constexpr = False,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    
    if use_tma:
        tensor_desc = tl.make_tensor_descriptor(
            tensor_ptr,
            [M, K],
            [stride_m_t, stride_k_t],
            [BLOCK_SIZE_M, GROUP_SIZE]
        )        
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            [M, K // 2],
            [stride_m_o, stride_k_o],
            [BLOCK_SIZE_M, HALF_GROUP_SIZE]
        )

    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    
    if use_tma:        
        tensor = tl.load_tensor_descriptor(tensor_desc, [pid_m * BLOCK_SIZE_M, pid_k * GROUP_SIZE]).to(tl.float32)
    else:
        tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    #next power of 2 via log
    scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor), axis=1, keep_dims=True) / 6., eps_exp)

    #Map to index
    wq = tensor / scales
    idx_abs = tl.sum(tl.abs(wq[:, :, None]) > thr_pos[None, :, :], axis=2)
    out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

    #Pack
    lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_GROUP_SIZE, 2), can_reorder=False))
    out = lo | (hi << 4)

    #Store
    offs_k = pid_k * HALF_GROUP_SIZE + tl.arange(0, HALF_GROUP_SIZE)
    out_mask = ((offs_m[:, None] < M) & (offs_k[None, :] < (K // 2))).to(tl.int1)
    if use_tma:
        tl.store_tensor_descriptor(out_desc, [pid_m * BLOCK_SIZE_M, pid_k * HALF_GROUP_SIZE], out)
    else:
        tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k[None, :] * stride_k_o), out, mask=out_mask)

    offs_k = pid_k * 1 + tl.arange(0, 1)
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales_log2)

def scale_activations_mxfp4_triton(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size: int = 32
    eps_exp: int = -30
    eps: float = 2 ** eps_exp

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)
        
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(K, group_size))
    device_index = tensor.device.index

    M_CLOSEST = get_closest_m(M)
    scale_activations_mxfp4_triton_kernel[grid](
        tensor,
        out,
        scales,
        thr_pos[device_index],
        M, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        #########################
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
    )

    return out, scales


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4, num_stages=3),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_nvfp4_triton_kernel(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    thr_pos_ptr,
    M, K, M_CLOSEST,
    #########################
    stride_m_t: tl.constexpr, 
    stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, 
    stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, 
    stride_k_o: tl.constexpr,
    #########################
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    meta_scales_ptr,
    use_tma: tl.constexpr = False,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    
    if use_tma:
        tensor_desc = tl.make_tensor_descriptor(
            tensor_ptr,
            [M, K],
            [stride_m_t, stride_k_t],
            [BLOCK_SIZE_M, GROUP_SIZE]
        )        
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            [M, K // 2],
            [stride_m_o, stride_k_o],
            [BLOCK_SIZE_M, HALF_GROUP_SIZE]
        )

    fp8_dtype: tl.constexpr = tl.float8e4nv
    max_fp8: tl.constexpr = 448.
    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]
    #thr_pos += 1e-6

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    
    if use_tma:
        tensor = tl.load_tensor_descriptor(tensor_desc, [pid_m * BLOCK_SIZE_M, pid_k * GROUP_SIZE]).to(tl.float32)
    else:
        tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)
        
    #FP8 scales
    meta_scales = tl.load(meta_scales_ptr, eviction_policy='evict_last')
    scales = tl.max(tl.abs(tensor), axis=1, keep_dims=True) * meta_scales / 6.
    scales = tl.minimum(scales, max_fp8).to(fp8_dtype)

    #Map to index
    scales_full = tl.maximum(scales.to(tl.float32) / meta_scales, eps)
    wq = tensor / scales_full
    idx_abs = tl.sum(tl.abs(wq[:, :, None]) > thr_pos[None, :, :], axis=2)
    out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

    #Pack
    lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_GROUP_SIZE, 2), can_reorder=False))
    out = lo | (hi << 4)

    #Store
    offs_k = pid_k * HALF_GROUP_SIZE + tl.arange(0, HALF_GROUP_SIZE)
    out_mask = ((offs_m[:, None] < M) & (offs_k[None, :] < (K // 2))).to(tl.int1)    
    if use_tma:
        tl.store_tensor_descriptor(out_desc, [pid_m * BLOCK_SIZE_M, pid_k * HALF_GROUP_SIZE], out)
    else:
        tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k[None, :] * stride_k_o), out, mask=out_mask)

    offs_k = pid_k + tl.arange(0, 1)
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales)


def scale_activations_nvfp4_triton(tensor: torch.Tensor, meta_scale=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size: int = 16
    eps: float = 1e-6
    fp8_dtype = torch.float8_e4m3fn #Nvidia only
    meta_scale = meta_scale if meta_scale is not None else (448.0 / (tensor.view(-1, 16).abs().amax(dim=1) / 6.0).max().clamp_(min=1e-6)).float()

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=fp8_dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(K, group_size))
    device_index = tensor.device.index

    M_CLOSEST = get_closest_m(M)
    scale_activations_nvfp4_triton_kernel[grid](
        tensor,
        out,
        scales,
        thr_pos[device_index],
        M, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        #########################
        eps=eps,
        GROUP_SIZE=group_size,
        meta_scales_ptr=meta_scale,
    )

    return out, scales, meta_scale

####################################################################################################################
# MXFP4 v2: persistent 1D grid, processes multiple K-groups per iteration
####################################################################################################################
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 8,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=1),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_mxfp4_triton_kernel_v2(
    tensor_ptr, out_ptr, scales_ptr, thr_pos_ptr,
    M, M_padded, K, M_CLOSEST,
    stride_m_t: tl.constexpr, stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, stride_k_o: tl.constexpr,
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_m_tiles = tl.cdiv(M_padded, BLOCK_SIZE_M)

    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
    HALF_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 2
    FLAT_M: tl.constexpr = BLOCK_SIZE_M * GROUPS_PER_BLOCK
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]

    for tile_m in range(pid, num_m_tiles, num_programs):
        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = offs_m < M

        tensor_bp = tl.make_block_ptr(
            tensor_ptr, (M, K), (stride_m_t, stride_k_t),
            (tile_m * BLOCK_SIZE_M, 0),
            (BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        out_bp = tl.make_block_ptr(
            out_ptr, (M, K // 2), (stride_m_o, stride_k_o),
            (tile_m * BLOCK_SIZE_M, 0),
            (BLOCK_SIZE_M, HALF_BLOCK_K), order=(1, 0)
        )

        for k_start in range(0, K, BLOCK_SIZE_K):
            tensor = tl.load(tensor_bp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

            # Reshape to [FLAT_M, GROUP_SIZE] for group-wise reduction
            tensor_flat = tl.reshape(tensor, (FLAT_M, GROUP_SIZE))

            # Per-group power-of-2 scale
            scales, scales_log2 = next_power_of_2_bitwise_triton(
                tl.max(tl.abs(tensor_flat), axis=1, keep_dims=True) / 6., eps_exp
            )

            # Map to FP4 index via threshold comparison
            wq = tensor_flat / scales
            idx_abs = tl.sum(tl.abs(wq[:, :, None]) > thr_pos[None, :, :], axis=2)
            out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

            # Reshape to [BLOCK_M, BLOCK_K] then pack pairs
            out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_K))
            lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False))
            out = lo | (hi << 4)

            tl.store(out_bp, out, boundary_check=(0, 1))

            # Store scales: [FLAT_M, 1] → [BLOCK_M, GROUPS_PER_BLOCK]
            scales_2d = tl.reshape(scales_log2, (BLOCK_SIZE_M, GROUPS_PER_BLOCK))
            group_idx = k_start // GROUP_SIZE
            offs_g = group_idx + tl.arange(0, GROUPS_PER_BLOCK)
            g_mask = offs_g < tl.cdiv(K, GROUP_SIZE)
            tl.store(
                scales_ptr + offs_m[:, None] * stride_m_s + offs_g[None, :] * stride_k_s,
                scales_2d, mask=m_mask[:, None] & g_mask[None, :]
            )

            tensor_bp = tl.advance(tensor_bp, (0, BLOCK_SIZE_K))
            out_bp = tl.advance(out_bp, (0, HALF_BLOCK_K))


def scale_activations_mxfp4_triton_v2(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size: int = 32
    eps_exp: int = -30

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)

    grid = lambda meta: (min(get_num_SMs(), triton.cdiv(M, meta['BLOCK_SIZE_M'])),)
    device_index = tensor.device.index

    M_CLOSEST = get_closest_m(M)
    scale_activations_mxfp4_triton_kernel_v2[grid](
        tensor, out, scales, thr_pos[device_index],
        M, M_padded, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
    )

    return out, scales


####################################################################################################################
# NVFP4 v2: persistent 1D grid, processes multiple K-groups per iteration
####################################################################################################################
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 8,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=1),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_nvfp4_triton_kernel_v2(
    tensor_ptr, out_ptr, scales_ptr, thr_pos_ptr,
    M, M_padded, K, M_CLOSEST,
    stride_m_t: tl.constexpr, stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, stride_k_o: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    meta_scales_ptr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_m_tiles = tl.cdiv(M_padded, BLOCK_SIZE_M)

    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
    HALF_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 2
    FLAT_M: tl.constexpr = BLOCK_SIZE_M * GROUPS_PER_BLOCK
    fp8_dtype: tl.constexpr = tl.float8e4nv
    max_fp8: tl.constexpr = 448.
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty
    thr_pos = tl.load(thr_pos_ptr + tl.arange(0, 8), eviction_policy='evict_last')[None, :]

    meta_scales = tl.load(meta_scales_ptr, eviction_policy='evict_last')
    for tile_m in range(pid, num_m_tiles, num_programs):
        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = offs_m < M

        tensor_bp = tl.make_block_ptr(
            tensor_ptr, (M, K), (stride_m_t, stride_k_t),
            (tile_m * BLOCK_SIZE_M, 0),
            (BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        out_bp = tl.make_block_ptr(
            out_ptr, (M, K // 2), (stride_m_o, stride_k_o),
            (tile_m * BLOCK_SIZE_M, 0),
            (BLOCK_SIZE_M, HALF_BLOCK_K), order=(1, 0)
        )

        for k_start in range(0, K, BLOCK_SIZE_K):
            tensor = tl.load(tensor_bp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

            # Reshape to [FLAT_M, GROUP_SIZE] for group-wise reduction
            tensor_flat = tl.reshape(tensor, (FLAT_M, GROUP_SIZE))

            # Per-group FP8 scale
            abs_max = tl.max(tl.abs(tensor_flat), axis=1, keep_dims=True)
            scales_raw = abs_max * meta_scales / 6.
            scales_fp8 = tl.minimum(scales_raw, max_fp8).to(fp8_dtype)
            scales_full = tl.maximum(scales_fp8.to(tl.float32) / meta_scales, eps)

            # Map to FP4 index via threshold comparison
            wq = tensor_flat / scales_full
            idx_abs = tl.sum(tl.abs(wq[:, :, None]) > thr_pos[None, :, :], axis=2)
            out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

            # Reshape to [BLOCK_M, BLOCK_K] then pack pairs
            out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_K))
            lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False))
            out = lo | (hi << 4)

            tl.store(out_bp, out, boundary_check=(0, 1))

            # Store scales: [FLAT_M, 1] → [BLOCK_M, GROUPS_PER_BLOCK]
            scales_2d = tl.reshape(scales_fp8, (BLOCK_SIZE_M, GROUPS_PER_BLOCK))
            group_idx = k_start // GROUP_SIZE
            offs_g = group_idx + tl.arange(0, GROUPS_PER_BLOCK)
            g_mask = offs_g < tl.cdiv(K, GROUP_SIZE)
            tl.store(
                scales_ptr + offs_m[:, None] * stride_m_s + offs_g[None, :] * stride_k_s,
                scales_2d, mask=m_mask[:, None] & g_mask[None, :]
            )

            tensor_bp = tl.advance(tensor_bp, (0, BLOCK_SIZE_K))
            out_bp = tl.advance(out_bp, (0, HALF_BLOCK_K))


def scale_activations_nvfp4_triton_v2(tensor: torch.Tensor, meta_scale=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size: int = 16
    eps: float = 1e-6
    fp8_dtype = torch.float8_e4m3fn
    meta_scale = meta_scale if meta_scale is not None else (448.0 / (tensor.view(-1, 16).abs().amax(dim=1) / 6.0).max().clamp_(min=1e-6)).float()

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=fp8_dtype)

    grid = lambda meta: (min(get_num_SMs(), triton.cdiv(M, meta['BLOCK_SIZE_M'])),)
    device_index = tensor.device.index

    M_CLOSEST = get_closest_m(M)
    scale_activations_nvfp4_triton_kernel_v2[grid](
        tensor, out, scales, thr_pos[device_index],
        M, M_padded, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        eps=eps,
        GROUP_SIZE=group_size,
        meta_scales_ptr=meta_scale,
    )

    return out, scales, meta_scale


####################################################################################################################
# MXFP4 v3: 2D grid like v1, but scalar threshold loop to avoid 3D tensor
####################################################################################################################
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64},  num_warps=4, num_stages=3),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_mxfp4_triton_kernel_v3(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    thr_pos_ptr,
    M, K, M_CLOSEST,
    #########################
    stride_m_t: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr,
    stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr,
    stride_k_o: tl.constexpr,
    #########################
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    use_tma: tl.constexpr = False,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    # Load 8 thresholds as individual scalars
    thr0 = tl.load(thr_pos_ptr + 0)
    thr1 = tl.load(thr_pos_ptr + 1)
    thr2 = tl.load(thr_pos_ptr + 2)
    thr3 = tl.load(thr_pos_ptr + 3)
    thr4 = tl.load(thr_pos_ptr + 4)
    thr5 = tl.load(thr_pos_ptr + 5)
    thr6 = tl.load(thr_pos_ptr + 6)
    thr7 = tl.load(thr_pos_ptr + 7)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)

    #next power of 2 via log
    scales, scales_log2 = next_power_of_2_triton(tl.max(tl.abs(tensor), axis=1, keep_dims=True) / 6., eps_exp)

    #Map to index via scalar threshold comparisons (avoids 3D intermediate)
    wq = tensor / scales
    abs_wq = tl.abs(wq)
    idx_abs = ((abs_wq > thr0).to(tl.int32) + (abs_wq > thr1).to(tl.int32) +
               (abs_wq > thr2).to(tl.int32) + (abs_wq > thr3).to(tl.int32) +
               (abs_wq > thr4).to(tl.int32) + (abs_wq > thr5).to(tl.int32) +
               (abs_wq > thr6).to(tl.int32) + (abs_wq > thr7).to(tl.int32))
    out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

    #Pack
    lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_GROUP_SIZE, 2), can_reorder=False))
    out = lo | (hi << 4)

    #Store
    offs_k = pid_k * HALF_GROUP_SIZE + tl.arange(0, HALF_GROUP_SIZE)
    out_mask = ((offs_m[:, None] < M) & (offs_k[None, :] < (K // 2))).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k[None, :] * stride_k_o), out, mask=out_mask)

    offs_k = pid_k * 1 + tl.arange(0, 1)
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales_log2)

def scale_activations_mxfp4_triton_v3(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size: int = 32
    eps_exp: int = -30

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(K, group_size))
    device_index = tensor.device.index

    M_CLOSEST = get_closest_m(M)
    scale_activations_mxfp4_triton_kernel_v3[grid](
        tensor,
        out,
        scales,
        thr_pos[device_index],
        M, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        #########################
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
    )

    return out, scales


####################################################################################################################
# NVFP4 v3: 2D grid like v1, but scalar threshold loop to avoid 3D tensor
####################################################################################################################
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64},  num_warps=4, num_stages=3),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_nvfp4_triton_kernel_v3(
    tensor_ptr,
    out_ptr,
    scales_ptr,
    thr_pos_ptr,
    M, K, M_CLOSEST,
    #########################
    stride_m_t: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr,
    stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr,
    stride_k_o: tl.constexpr,
    #########################
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    meta_scales_ptr,
    use_tma: tl.constexpr = False,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    fp8_dtype: tl.constexpr = tl.float8e4nv
    max_fp8: tl.constexpr = 448.
    HALF_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    # Load 8 thresholds as individual scalars
    thr0 = tl.load(thr_pos_ptr + 0)
    thr1 = tl.load(thr_pos_ptr + 1)
    thr2 = tl.load(thr_pos_ptr + 2)
    thr3 = tl.load(thr_pos_ptr + 3)
    thr4 = tl.load(thr_pos_ptr + 4)
    thr5 = tl.load(thr_pos_ptr + 5)
    thr6 = tl.load(thr_pos_ptr + 6)
    thr7 = tl.load(thr_pos_ptr + 7)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    #Load
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)

    #FP8 scales
    meta_scales = tl.load(meta_scales_ptr, eviction_policy='evict_last')
    scales = tl.max(tl.abs(tensor), axis=1, keep_dims=True) * meta_scales / 6.
    scales = tl.minimum(scales, max_fp8).to(fp8_dtype)

    #Map to index via scalar threshold comparisons (avoids 3D intermediate)
    scales_full = tl.maximum(scales.to(tl.float32) / meta_scales, eps)
    wq = tensor / scales_full
    abs_wq = tl.abs(wq)
    idx_abs = ((abs_wq > thr0).to(tl.int32) + (abs_wq > thr1).to(tl.int32) +
               (abs_wq > thr2).to(tl.int32) + (abs_wq > thr3).to(tl.int32) +
               (abs_wq > thr4).to(tl.int32) + (abs_wq > thr5).to(tl.int32) +
               (abs_wq > thr6).to(tl.int32) + (abs_wq > thr7).to(tl.int32))
    out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

    #Pack
    lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_GROUP_SIZE, 2), can_reorder=False))
    out = lo | (hi << 4)

    #Store
    offs_k = pid_k * HALF_GROUP_SIZE + tl.arange(0, HALF_GROUP_SIZE)
    out_mask = ((offs_m[:, None] < M) & (offs_k[None, :] < (K // 2))).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k[None, :] * stride_k_o), out, mask=out_mask)

    offs_k = pid_k + tl.arange(0, 1)
    tl.store(scales_ptr + (offs_m[:, None] * stride_m_s + offs_k[None, :] * stride_k_s), scales)


def scale_activations_nvfp4_triton_v3(tensor: torch.Tensor, meta_scale=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size: int = 16
    eps: float = 1e-6
    fp8_dtype = torch.float8_e4m3fn
    meta_scale = meta_scale if meta_scale is not None else (448.0 / (tensor.view(-1, 16).abs().amax(dim=1) / 6.0).max().clamp_(min=1e-6)).float()

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=fp8_dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(K, group_size))
    device_index = tensor.device.index

    M_CLOSEST = get_closest_m(M)
    scale_activations_nvfp4_triton_kernel_v3[grid](
        tensor,
        out,
        scales,
        thr_pos[device_index],
        M, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        #########################
        eps=eps,
        GROUP_SIZE=group_size,
        meta_scales_ptr=meta_scale,
    )

    return out, scales, meta_scale



####################################################################################################################
# MXFP4 v5: 2D grid with multi-group BLOCK_SIZE_K (fewer blocks, better bandwidth)
####################################################################################################################
def prune_large_blocks_2d(configs, named_args, **kwargs):
    M = named_args.get('M_CLOSEST', named_args.get('M'))
    K = named_args['K']

    pruned = []
    for config in configs:
        bm = config.kwargs['BLOCK_SIZE_M']
        bk = config.kwargs['BLOCK_SIZE_K']
        if bm <= M and bk <= K:
            pruned.append(config)

    if not pruned:
        pruned.append(configs[0])

    return pruned

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 256}, num_warps=8, num_stages=1),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks_2d},
)
@triton.jit
def scale_activations_mxfp4_triton_kernel_v5(
    tensor_ptr, out_ptr, scales_ptr, thr_pos_ptr,
    M, M_padded, K, M_CLOSEST,
    stride_m_t: tl.constexpr, stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, stride_k_o: tl.constexpr,
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Requires CUDA 13.0+ ptxas (Triton bundles 12.9 as of v3.3). To enable, replace
    # the bundled ptxas-blackwell with the system one: cp /usr/local/cuda/bin/ptxas
    # /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas-blackwell
    # TODO: once Triton ships CUDA 13.0+ ptxas, set default to True and add ptx_pack
    # to the autotuner configs so it can pick the best path per shape.
    ptx_pack: tl.constexpr = GEMLITE_ENABLE_PTX_FP4_PACK,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    HALF_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 2
    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
    FLAT_M: tl.constexpr = BLOCK_SIZE_M * GROUPS_PER_BLOCK
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)

    tensor_flat = tl.reshape(tensor, (FLAT_M, GROUP_SIZE))

    scales, scales_log2 = next_power_of_2_triton(
        tl.max(tl.abs(tensor_flat), axis=1, keep_dims=True) / 6., eps_exp
    )

    wq = tensor_flat / scales

    if ptx_pack:
        # PTX path: hardware e2m1x2 quantization + nibble packing
        wq_2d = tl.reshape(wq, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        wq_pairs = wq_2d.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False)
        lo_val, hi_val = tl.split(wq_pairs)
        lo_f16 = lo_val.to(tl.float16)
        hi_f16 = hi_val.to(tl.float16)
        lo_bits = lo_f16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        hi_bits = (hi_f16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) << 16
        packed_f16x2 = lo_bits | hi_bits
        packed_e2m1 = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b8       tmp_out;
                .reg .f16x2    tmp_in;
                mov.b32                          tmp_in, $1;
                cvt.rn.satfinite.e2m1x2.f16x2    tmp_out, tmp_in;
                cvt.u32.u8                       $0, tmp_out;
            }
            """,
            constraints="=r,r",
            args=[packed_f16x2],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        out = packed_e2m1.to(tl.uint8)
    else:
        # Threshold path: 8 comparisons + manual nibble packing
        thr0 = tl.load(thr_pos_ptr + 0)
        thr1 = tl.load(thr_pos_ptr + 1)
        thr2 = tl.load(thr_pos_ptr + 2)
        thr3 = tl.load(thr_pos_ptr + 3)
        thr4 = tl.load(thr_pos_ptr + 4)
        thr5 = tl.load(thr_pos_ptr + 5)
        thr6 = tl.load(thr_pos_ptr + 6)
        thr7 = tl.load(thr_pos_ptr + 7)
        abs_wq = tl.abs(wq)
        idx_abs = ((abs_wq > thr0).to(tl.int32) + (abs_wq > thr1).to(tl.int32) +
                   (abs_wq > thr2).to(tl.int32) + (abs_wq > thr3).to(tl.int32) +
                   (abs_wq > thr4).to(tl.int32) + (abs_wq > thr5).to(tl.int32) +
                   (abs_wq > thr6).to(tl.int32) + (abs_wq > thr7).to(tl.int32))
        out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)
        out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False))
        out = lo | (hi << 4)

    offs_k_out = pid_k * HALF_BLOCK_K + tl.arange(0, HALF_BLOCK_K)
    out_mask = ((offs_m[:, None] < M) & (offs_k_out[None, :] < (K // 2))).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k_out[None, :] * stride_k_o), out, mask=out_mask)

    scales_2d = tl.reshape(scales_log2, (BLOCK_SIZE_M, GROUPS_PER_BLOCK))
    # For padding rows (M <= row < M_padded), store identity scale (127 = 2^0 in E8M0)
    scales_2d = tl.where(offs_m[:, None] < M, scales_2d, tl.full(scales_2d.shape, 127, dtype=tl.uint8))
    base_group = pid_k * GROUPS_PER_BLOCK
    offs_g = base_group + tl.arange(0, GROUPS_PER_BLOCK)
    g_mask = offs_g < tl.cdiv(K, GROUP_SIZE)
    tl.store(
        scales_ptr + offs_m[:, None] * stride_m_s + offs_g[None, :] * stride_k_s,
        scales_2d, mask=(offs_m[:, None] < M_padded) & g_mask[None, :]
    )


def scale_activations_mxfp4_triton_v5(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    group_size: int = 32
    eps_exp: int = -30

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)

    grid = lambda meta: (triton.cdiv(M_padded, meta['BLOCK_SIZE_M']), triton.cdiv(K, meta['BLOCK_SIZE_K']))
    device_index = tensor.device.index

    M_CLOSEST = get_closest_m(M_padded)
    scale_activations_mxfp4_triton_kernel_v5[grid](
        tensor, out, scales, thr_pos[device_index],
        M, M_padded, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
    )
    return out, scales


####################################################################################################################
# Per-device buffers for dynamic NVFP4 meta_scale computation.
# Allocated lazily on first use (NEVER at import , import-time cudaMalloc
# crashes under the GreenBoost LD_PRELOAD shim). LazyDeviceTensors caches the
# buffers, so their addresses are stable across CUDAGraph replays and it
# refuses to allocate during capture , run a warmup call before capturing.
_nvfp4_meta_scale_bufs = LazyDeviceTensors([0.0], torch.float32)  # meta_scale output (float32 scalar)
_nvfp4_amax_bufs = LazyDeviceTensors([0.0], torch.float32)        # atomic max scratch (float32 scalar)
_nvfp4_counter_bufs = LazyDeviceTensors([0], torch.int32)         # grid sync counter (int32 scalar)

def _get_nvfp4_bufs(device_index):
    return _nvfp4_meta_scale_bufs[device_index], _nvfp4_amax_bufs[device_index], _nvfp4_counter_bufs[device_index]

####################################################################################################################
# Fused persistent NVFP4 v6: Single-kernel amax + quantize
# Phase 1: all blocks compute tile amax, atomicMax to global, grid barrier
# Phase 2: all blocks quantize tiles using computed meta_scale
# Grid limited to num_SMs so all blocks run concurrently (spin-wait safe)
####################################################################################################################
@triton.jit
def scale_activations_nvfp4_fused_kernel_v6(
    tensor_ptr, out_ptr, scales_ptr, thr_pos_ptr,
    M, M_padded, K,
    stride_m_t: tl.constexpr, stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, stride_k_o: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    meta_scales_ptr,  # output: computed meta_scale
    amax_ptr,         # scratch: atomic max accumulator
    counter_ptr,      # scratch: grid sync counter
    num_tiles_m, num_tiles_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ptx_pack: tl.constexpr = False,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    total_tiles = num_tiles_m * num_tiles_k

    fp8_dtype: tl.constexpr = tl.float8e4nv
    max_fp8: tl.constexpr = 448.
    HALF_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 2
    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
    FLAT_M: tl.constexpr = BLOCK_SIZE_M * GROUPS_PER_BLOCK
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    # Load thresholds once
    thr0 = tl.load(thr_pos_ptr + 0)
    thr1 = tl.load(thr_pos_ptr + 1)
    thr2 = tl.load(thr_pos_ptr + 2)
    thr3 = tl.load(thr_pos_ptr + 3)
    thr4 = tl.load(thr_pos_ptr + 4)
    thr5 = tl.load(thr_pos_ptr + 5)
    thr6 = tl.load(thr_pos_ptr + 6)
    thr7 = tl.load(thr_pos_ptr + 7)

    # ---- Phase 1: Compute amax across all tiles ----
    local_amax = tl.full((1,), value=0.0, dtype=tl.float32)
    for tile_idx in range(pid, total_tiles, num_pids):
        tile_m = tile_idx // num_tiles_k
        tile_k = tile_idx % num_tiles_k

        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_k = tile_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
        tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
        tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)

        tile_max = tl.max(tl.abs(tensor))
        local_amax = tl.maximum(local_amax, tile_max)

    # Atomic max to global (release: ensures atomicMax is visible before counter increment)
    tl.atomic_max(amax_ptr, tl.max(local_amax, axis=0), sem='relaxed')

    # Grid barrier: last block computes meta_scale and signals
    # acq_rel: acquires all prior releases (sees all other blocks' atomicMax)
    old_count = tl.atomic_add(counter_ptr, 1, sem='relaxed')
    if old_count == num_pids - 1:
        final_amax = tl.load(amax_ptr)
        tl.store(meta_scales_ptr, max_fp8 * 6.0 / tl.maximum(final_amax, eps))
        # Reset scratch for next call
        tl.store(amax_ptr, 0.0)
        # Signal ready by setting counter to -num_pids (distinguishable from 0..num_pids-1)
        tl.store(counter_ptr, -1)

    # Spin-wait for ready signal (safe: grid <= num_SMs, all blocks run concurrently)
    while tl.atomic_add(counter_ptr, 0, sem='relaxed') >= 0:
        pass

    # ---- Phase 2: Quantize using computed meta_scale ----
    meta_scales = tl.load(meta_scales_ptr)

    for tile_idx in range(pid, total_tiles, num_pids):
        tile_m = tile_idx // num_tiles_k
        tile_k = tile_idx % num_tiles_k

        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_k = tile_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Reload tile (L2 cached from Phase 1)
        mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
        tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
        tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)

        tensor_flat = tl.reshape(tensor, (FLAT_M, GROUP_SIZE))
        abs_max = tl.max(tl.abs(tensor_flat), axis=1, keep_dims=True)
        scales_raw = abs_max * meta_scales / 6.
        scales_fp8 = tl.minimum(scales_raw, max_fp8).to(fp8_dtype)
        scales_full = tl.maximum(scales_fp8.to(tl.float32) / meta_scales, eps)

        wq = tensor_flat / scales_full

        if ptx_pack:
            wq_2d = tl.reshape(wq, (BLOCK_SIZE_M, BLOCK_SIZE_K))
            wq_pairs = wq_2d.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False)
            lo_val, hi_val = tl.split(wq_pairs)
            lo_f16 = lo_val.to(tl.float16)
            hi_f16 = hi_val.to(tl.float16)
            lo_bits = lo_f16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            hi_bits = (hi_f16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) << 16
            packed_f16x2 = lo_bits | hi_bits
            packed_e2m1 = tl.inline_asm_elementwise(
                asm="""
                {
                    .reg .b8       tmp_out;
                    .reg .f16x2    tmp_in;
                    mov.b32                          tmp_in, $1;
                    cvt.rn.satfinite.e2m1x2.f16x2    tmp_out, tmp_in;
                    cvt.u32.u8                       $0, tmp_out;
                }
                """,
                constraints="=r,r",
                args=[packed_f16x2],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            out = packed_e2m1.to(tl.uint8)
        else:
            abs_wq = tl.abs(wq)
            idx_abs = ((abs_wq > thr0).to(tl.int32) + (abs_wq > thr1).to(tl.int32) +
                       (abs_wq > thr2).to(tl.int32) + (abs_wq > thr3).to(tl.int32) +
                       (abs_wq > thr4).to(tl.int32) + (abs_wq > thr5).to(tl.int32) +
                       (abs_wq > thr6).to(tl.int32) + (abs_wq > thr7).to(tl.int32))
            out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)
            out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_K))
            lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False))
            out = lo | (hi << 4)

        # Store quantized output
        offs_k_out = tile_k * HALF_BLOCK_K + tl.arange(0, HALF_BLOCK_K)
        out_mask = ((offs_m[:, None] < M) & (offs_k_out[None, :] < (K // 2))).to(tl.int1)
        tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k_out[None, :] * stride_k_o), out, mask=out_mask)

        # Store scales
        scales_2d = tl.reshape(scales_fp8, (BLOCK_SIZE_M, GROUPS_PER_BLOCK))
        scales_2d = tl.where(offs_m[:, None] < M, scales_2d, tl.full(scales_2d.shape, 1.0, dtype=tl.float32).to(fp8_dtype))
        base_group = tile_k * GROUPS_PER_BLOCK
        offs_g = base_group + tl.arange(0, GROUPS_PER_BLOCK)
        g_mask = offs_g < tl.cdiv(K, GROUP_SIZE)
        tl.store(
            scales_ptr + offs_m[:, None] * stride_m_s + offs_g[None, :] * stride_k_s,
            scales_2d, mask=(offs_m[:, None] < M_padded) & g_mask[None, :]
        )

    # Last block resets counter for next call
    if old_count == num_pids - 1:
        tl.store(counter_ptr, 0)

####################################################################################################################
# NVFP4 v5: 2D grid with multi-group BLOCK_SIZE_K (fewer blocks, better bandwidth)
####################################################################################################################
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 16},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 256}, num_warps=8, num_stages=1),
    ],
    key=['M_CLOSEST', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks_2d},
)
@triton.jit
def scale_activations_nvfp4_triton_kernel_v5(
    tensor_ptr, out_ptr, scales_ptr, thr_pos_ptr,
    M, M_padded, K, M_CLOSEST,
    stride_m_t: tl.constexpr, stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, stride_k_o: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    meta_scales_ptr,
    # Requires CUDA 13.0+ ptxas (Triton bundles 12.9 as of v3.3). To enable, set
    # the environment variable TRITON_CUDA_ARCH_LIST to include CUDA 13.0+ ptxas, 
    # and override the bundled ptxas-blackwell.
    ptx_pack: tl.constexpr = False,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    fp8_dtype: tl.constexpr = tl.float8e4nv
    max_fp8: tl.constexpr = 448.
    HALF_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 2
    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
    FLAT_M: tl.constexpr = BLOCK_SIZE_M * GROUPS_PER_BLOCK
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)
    meta_scales = tl.load(meta_scales_ptr, eviction_policy='evict_last')

    tensor_flat = tl.reshape(tensor, (FLAT_M, GROUP_SIZE))
    abs_max = tl.max(tl.abs(tensor_flat), axis=1, keep_dims=True)
    scales_raw = abs_max * meta_scales / 6.
    scales_fp8 = tl.minimum(scales_raw, max_fp8).to(fp8_dtype)
    scales_full = tl.maximum(scales_fp8.to(tl.float32) / meta_scales, eps)

    wq = tensor_flat / scales_full

    if ptx_pack:
        # PTX path: hardware e2m1x2 quantization + nibble packing
        wq_2d = tl.reshape(wq, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        wq_pairs = wq_2d.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False)
        lo_val, hi_val = tl.split(wq_pairs)
        lo_f16 = lo_val.to(tl.float16)
        hi_f16 = hi_val.to(tl.float16)
        lo_bits = lo_f16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        hi_bits = (hi_f16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) << 16
        packed_f16x2 = lo_bits | hi_bits
        packed_e2m1 = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b8       tmp_out;
                .reg .f16x2    tmp_in;
                mov.b32                          tmp_in, $1;
                cvt.rn.satfinite.e2m1x2.f16x2    tmp_out, tmp_in;
                cvt.u32.u8                       $0, tmp_out;
            }
            """,
            constraints="=r,r",
            args=[packed_f16x2],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        out = packed_e2m1.to(tl.uint8)
    else:
        # Threshold path: 8 comparisons + manual nibble packing
        thr0 = tl.load(thr_pos_ptr + 0)
        thr1 = tl.load(thr_pos_ptr + 1)
        thr2 = tl.load(thr_pos_ptr + 2)
        thr3 = tl.load(thr_pos_ptr + 3)
        thr4 = tl.load(thr_pos_ptr + 4)
        thr5 = tl.load(thr_pos_ptr + 5)
        thr6 = tl.load(thr_pos_ptr + 6)
        thr7 = tl.load(thr_pos_ptr + 7)
        abs_wq = tl.abs(wq)
        idx_abs = ((abs_wq > thr0).to(tl.int32) + (abs_wq > thr1).to(tl.int32) +
                   (abs_wq > thr2).to(tl.int32) + (abs_wq > thr3).to(tl.int32) +
                   (abs_wq > thr4).to(tl.int32) + (abs_wq > thr5).to(tl.int32) +
                   (abs_wq > thr6).to(tl.int32) + (abs_wq > thr7).to(tl.int32))
        out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)
        out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False))
        out = lo | (hi << 4)

    offs_k_out = pid_k * HALF_BLOCK_K + tl.arange(0, HALF_BLOCK_K)
    out_mask = ((offs_m[:, None] < M) & (offs_k_out[None, :] < (K // 2))).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k_out[None, :] * stride_k_o), out, mask=out_mask)

    scales_2d = tl.reshape(scales_fp8, (BLOCK_SIZE_M, GROUPS_PER_BLOCK))
    # For padding rows (M <= row < M_padded), store identity scale (1.0 in fp8)
    scales_2d = tl.where(offs_m[:, None] < M, scales_2d, tl.full(scales_2d.shape, 1.0, dtype=tl.float32).to(fp8_dtype))
    base_group = pid_k * GROUPS_PER_BLOCK
    offs_g = base_group + tl.arange(0, GROUPS_PER_BLOCK)
    g_mask = offs_g < tl.cdiv(K, GROUP_SIZE)
    tl.store(
        scales_ptr + offs_m[:, None] * stride_m_s + offs_g[None, :] * stride_k_s,
        scales_2d, mask=(offs_m[:, None] < M_padded) & g_mask[None, :]
    )


def scale_activations_nvfp4_triton_v5(tensor: torch.Tensor, meta_scale=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size: int = 16
    eps: float = 1e-6
    fp8_dtype = torch.float8_e4m3fn

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=fp8_dtype)
    device_index = tensor.device.index

    if meta_scale is None:
        # Fused path: single kernel computes amax + quantizes
        meta_scale, amax_buf, counter_buf = _get_nvfp4_bufs(device_index)
        BLOCK_M = 32 if M >= 32 else 16
        BLOCK_K = 256
        num_tiles_m = triton.cdiv(M_padded, BLOCK_M)
        num_tiles_k = triton.cdiv(K, BLOCK_K)
        num_SMs = torch.cuda.get_device_properties(device_index).multi_processor_count
        num_blocks = min(num_tiles_m * num_tiles_k, num_SMs)

        scale_activations_nvfp4_fused_kernel_v6[(num_blocks,)](
            tensor, out, scales, thr_pos[device_index],
            M, M_padded, K,
            tensor.stride(0), tensor.stride(1),
            scales.stride(0), scales.stride(1),
            out.stride(0), out.stride(1),
            eps=eps,
            GROUP_SIZE=group_size,
            meta_scales_ptr=meta_scale,
            amax_ptr=amax_buf,
            counter_ptr=counter_buf,
            num_tiles_m=num_tiles_m,
            num_tiles_k=num_tiles_k,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_K=BLOCK_K,
            ptx_pack=GEMLITE_ENABLE_PTX_FP4_PACK,
        )
    else:
        # Static path: meta_scale already provided, use v5 kernel directly
        M_CLOSEST = get_closest_m(M_padded)
        grid = lambda meta: (triton.cdiv(M_padded, meta['BLOCK_SIZE_M']), triton.cdiv(K, meta['BLOCK_SIZE_K']))
        scale_activations_nvfp4_triton_kernel_v5[grid](
            tensor, out, scales, thr_pos[device_index],
            M, M_padded, K, M_CLOSEST,
            tensor.stride(0), tensor.stride(1),
            scales.stride(0), scales.stride(1),
            out.stride(0), out.stride(1),
            eps=eps,
            GROUP_SIZE=group_size,
            meta_scales_ptr=meta_scale,
            ptx_pack=GEMLITE_ENABLE_PTX_FP4_PACK,
        )

    return out, scales, meta_scale




####################################################################################################################
#INT8 / FP8 per-block activations (block quantization)
####################################################################################################################
from .triton_kernels.config import BLOCK_QUANT_SIZE

@torch.compile(fullgraph=True)
def scale_activations_per_block_torch(
    tensor: Tensor, w_dtype: torch.dtype, block_size: int = BLOCK_QUANT_SIZE,
) -> Tuple[Tensor, Tensor]:
    min_val, max_val = get_dtype_range(w_dtype)
    out_shape = tensor.shape
    tensor = tensor.to(torch.float32, copy=False).view(-1, tensor.shape[-1])
    M, K = tensor.shape
    assert K % block_size == 0, "K must be divisible by block_size for per-block activation quantization."

    # [M, K//B, B] -> amax over B
    t = tensor.view(M, K // block_size, block_size)
    scales = t.abs().amax(dim=-1) / max_val  # [M, K//B]
    scales = scales.clamp_(min=1e-6)
    out = t / scales.unsqueeze(-1)
    out = out.clamp_(min_val, max_val)
    if not w_dtype.is_floating_point:
        out = out.round_()
    out = out.to(dtype=w_dtype).view(out_shape)
    return out, scales.contiguous()


@triton.jit
def scale_activations_per_block_triton_v1_kernel(
    tensor_ptr, scale_ptr, y_ptr,
    M, K,
    stride_m: tl.constexpr,
    stride_k: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sg: tl.constexpr,
    ROUND: tl.constexpr,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == block_size (quantization block)
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    m_mask = offs_m < M
    mask = m_mask[:, None] & (offs_k[None, :] < K)

    offsets = offs_m[:, None] * stride_m + offs_k[None, :] * stride_k
    tensor = tl.load(tensor_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    scales_x = tl.max(tl.abs(tensor), axis=1) / max_val
    scales_x = tl.maximum(scales_x, 1e-6)
    tensor = tensor / scales_x[:, None]
    tensor = tl.minimum(tl.maximum(tensor, min_val), max_val)
    if ROUND:
        tensor = round_triton(tensor)

    tl.store(scale_ptr + offs_m * stride_sm + pid_k * stride_sg, scales_x, mask=m_mask)
    tl.store(y_ptr + offsets, tensor, mask=mask)


def scale_activations_per_block_triton_v1(
    tensor: Tensor, w_dtype: torch.dtype, block_size: int = BLOCK_QUANT_SIZE,
) -> Tuple[Tensor, Tensor]:
    min_val, max_val = get_dtype_range(w_dtype)
    x_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape
    assert K % block_size == 0, "K must be divisible by block_size for per-block activation quantization."

    num_k_blocks = K // block_size
    scales = torch.empty((M, num_k_blocks), dtype=torch.float32, device=tensor.device)
    y = torch.empty((M, K), dtype=w_dtype, device=tensor.device)

    BLOCK_SIZE_M = 1
    grid = (triton.cdiv(M, BLOCK_SIZE_M), num_k_blocks)
    ROUND = not w_dtype.is_floating_point

    scale_activations_per_block_triton_v1_kernel[grid](
        tensor, scales, y,
        M, K,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        ROUND=ROUND,
        min_val=min_val, max_val=max_val,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=block_size,
        num_stages=1, num_warps=4,
    )

    return y.view(x_shape), scales


####################################################################################################################
# v2: 2D grid, multi-row + multi-group per program via flat reshape.
# Each tile covers BLOCK_SIZE_M rows × BLOCK_SIZE_K cols (BLOCK_SIZE_K is a multiple of the 128 quant
# block); the tile is reshaped to [BLOCK_SIZE_M * GROUPS_PER_BLOCK, 128] so the per-128 amax becomes
# one row-reduction, letting each program amortize launch overhead over many rows.
# Autotune key is (M_CLOSEST, K), identical to the other quant_utils kernels , M_CLOSEST is bucketed
# by get_closest_m so changing M across buckets does not force a recompile.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1,  'BLOCK_SIZE_K': 128},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 1,  'BLOCK_SIZE_K': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 2,  'BLOCK_SIZE_K': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_K': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 8,  'BLOCK_SIZE_K': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_K': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 8,  'BLOCK_SIZE_K': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 512},  num_warps=8, num_stages=2),
    ],
    key=['M_CLOSEST', 'K'],
)
@triton.jit
def scale_activations_per_block_triton_v2_kernel(
    tensor_ptr, scale_ptr, y_ptr,
    M, K, M_CLOSEST,
    stride_m: tl.constexpr, stride_k: tl.constexpr,
    stride_sm: tl.constexpr, stride_sg: tl.constexpr,
    ROUND: tl.constexpr,
    min_val: tl.constexpr, max_val: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
    FLAT_M: tl.constexpr = BLOCK_SIZE_M * GROUPS_PER_BLOCK

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    m_mask = offs_m < M
    mask = m_mask[:, None] & (offs_k[None, :] < K)

    offsets = offs_m[:, None] * stride_m + offs_k[None, :] * stride_k
    x = tl.load(tensor_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x_flat = tl.reshape(x, (FLAT_M, GROUP_SIZE))
    amax = tl.max(tl.abs(x_flat), axis=1)
    scales_x = tl.maximum(amax / max_val, 1e-6)
    xq = x_flat * (1.0 / scales_x[:, None])
    xq = tl.clamp(xq, min_val, max_val)
    if ROUND:
        xq = round_triton(xq)

    xq = tl.reshape(xq, (BLOCK_SIZE_M, BLOCK_SIZE_K)).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offsets, xq, mask=mask)

    scales_2d = tl.reshape(scales_x, (BLOCK_SIZE_M, GROUPS_PER_BLOCK))
    offs_g = pid_k * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)
    mask = m_mask[:, None] & (offs_g < tl.cdiv(K, GROUP_SIZE))[None, :]
    tl.store(scale_ptr + offs_m[:, None] * stride_sm + offs_g[None, :] * stride_sg, scales_2d, mask=mask)


def scale_activations_per_block_triton_v2(
    tensor: Tensor, w_dtype: torch.dtype, block_size: int = BLOCK_QUANT_SIZE,
) -> Tuple[Tensor, Tensor]:
    min_val, max_val = get_dtype_range(w_dtype)
    x_shape = tensor.shape
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape
    assert K % block_size == 0, "K must be divisible by block_size for per-block activation quantization."

    scales = torch.empty((M, K // block_size), dtype=torch.float32, device=tensor.device)
    y = torch.empty((M, K), dtype=w_dtype, device=tensor.device)
    M_CLOSEST = get_closest_m(M)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(K, META['BLOCK_SIZE_K']))
    scale_activations_per_block_triton_v2_kernel[grid](
        tensor, scales, y,
        M, K, M_CLOSEST,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        ROUND=not w_dtype.is_floating_point,
        min_val=min_val, max_val=max_val,
        GROUP_SIZE=block_size,
    )
    return y.view(x_shape), scales


####################################################################################################################
scale_activations_per_token = scale_activations_per_token_triton_v3
scale_activations_per_block = scale_activations_per_block_triton_v2
scale_activations_mxfp8 = scale_activations_mxfp8_triton_v4
scale_activations_mxfp4 = scale_activations_mxfp4_triton_v5
scale_activations_nvfp4 = scale_activations_nvfp4_triton_v5
