# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

import torch, triton, math
import triton.language as tl
from triton.runtime import driver
from ..dtypes import *

GEMLITE_USE_NATIVE_ATOMIC_BFP16 = True

# Importing gemlite must NEVER touch the GPU. Eager import-time CUDA work
# (context init, cudaMalloc) crashes under the GreenBoost LD_PRELOAD shim
# (cudaErrorInvalidResourceHandle before the shim is ready) and breaks
# CPU-only imports. Device properties and per-device constant workspaces
# are therefore resolved lazily, on first use.
_GPU_COMPUTE_CAPABILITY = None


def gpu_compute_capability() -> int:
    global _GPU_COMPUTE_CAPABILITY
    if _GPU_COMPUTE_CAPABILITY is None:
        _GPU_COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]
    return _GPU_COMPUTE_CAPABILITY


class LazyDeviceTensors:
    """Per-device constant/scratch tensors allocated on FIRST USE.

    Drop-in replacement for the old import-time `list-per-device` pattern:
    call sites keep indexing with `[device_index]`. Buffers are cached, so
    addresses stay stable across CUDA-graph replays; allocation during graph
    capture is refused — run a warmup call first."""

    def __init__(self, values, dtype):
        self._values = values
        self._dtype = dtype
        self._bufs: dict = {}

    def __getitem__(self, device_index: int) -> torch.Tensor:
        buf = self._bufs.get(device_index)
        if buf is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "gemlite device workspace first requested during CUDA graph "
                    "capture — run a warmup call before capturing."
                )
            buf = torch.tensor(self._values, dtype=self._dtype,
                               device=f"cuda:{device_index}")
            self._bufs[device_index] = buf
        return buf

# TMA descriptors require a global memory allocation
from typing import Optional
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc_fn)


@triton.jit
def load_ptr(ptrs, mask, eviction_policy, apply_mask: tl.constexpr, other=0.):
    if apply_mask:
        return tl.load(ptrs, mask=mask, other=other, eviction_policy=eviction_policy)
    else:
        return tl.load(ptrs, eviction_policy=eviction_policy)

@triton.jit
def swizzle_tile_v1(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m     = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n     = tl.cdiv(N, BLOCK_SIZE_N)
    width      = GROUP_SIZE_M * grid_n
    group_id   = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m      = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n      = (pid % width) // group_size
    return pid_m, pid_n

@triton.jit
def swizzle_tile_v2(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m     = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n     = tl.cdiv(N, BLOCK_SIZE_N)
    width      = GROUP_SIZE_M * grid_m
    group_id   = pid // width
    group_size = tl.minimum(grid_n - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_n      = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_m      = (pid % width) // group_size
    return pid_m, pid_n

@triton.jit
def swizzle_tile_v3(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m  = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n  = pid // tl.cdiv(M, BLOCK_SIZE_M)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    return tl.swizzle2d(pid_m, pid_n, grid_m, grid_n, GROUP_SIZE_M)

@triton.jit
def swizzle_tile_persistent(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr): 
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

swizzle_tile = swizzle_tile_v1

@triton.jit
def linear_tile(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid // tl.cdiv(M, BLOCK_SIZE_M)
    return pid_m, pid_n

@triton.jit
def dequantize(
    b,
    scales,
    zeros,
    q_shift,
    meta_dtype,
    unpack_mask,
    elements_per_sample: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
):
    #Unpack
    if(elements_per_sample > 1):
        b = (b >> q_shift) & unpack_mask # int32 -> int32

    if(W_group_mode == 1): #Shift
        b = b.to(meta_dtype) - zeros 

    if(W_group_mode == 2):
        b = b.to(meta_dtype) * scales #Symmetric no shift (Grouped)

    if(W_group_mode == 3): #Asymmetric / Symmetric with shift(Grouped - (b - zeros) * scales)
        #b = (b - zeros) * scales
        if(zero_is_scalar):
            b = (b - zeros).to(meta_dtype) * scales
        else:
            b = (b.to(meta_dtype) - zeros) * scales

    if(W_group_mode == 4):
        b = tl.fma(b.to(meta_dtype), scales, zeros) #Asymmetric (Grouped - b*scales + zeros)

    return b

@triton.jit
def atomic_add_cas(ptr, value, Lock, mask=None, sem: tl.constexpr = "release"):    
    while tl.atomic_cas(Lock, 0, 1, sem=sem) == 1:
        pass
    tl.store(ptr, tl.load(ptr, mask=mask) + value, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

def next_power_of_2(v):
    return 2 ** int(math.ceil(math.log2(v)))

def is_divisible(dividend, divisor):
    return dividend % divisor == 0

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def gpu_has_more_shared_memory(
    ref_gpus=(
        "a100",
        "h100", "h200", "h20", "h800",
        "b100", "b200", "b300",
        "6000",
    ),
):
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return any(g in gpu_name for g in ref_gpus)

def gpu_supports_float16_acc(
    ref_gpus=["5090", "5080", "5070", "5060", 
              "4090", "4080", "4070", "4060",
              "3090", "3080", "3070", "3060",
              "2080", "2070"]
):
    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    return True in [g in gpu_name for g in ref_gpus]

def estimate_shared_memory_per_block(block_size_m, block_size_n, block_size_k, a_sizeof, b_sizeof, num_stages, e, g, load_scales_as_block):
    a_smem = block_size_m * block_size_k * a_sizeof
    if load_scales_as_block:
        # MX kernels: dot_scaled handles scaling natively, no dequant buffer
        # A tile: packed elements (e.g. NVFP4 packs 2 per byte, so K_A = K // e)
        a_smem = block_size_m * (block_size_k // e) * a_sizeof
        b_smem = (block_size_k // e) * block_size_n * b_sizeof
        # scales_b: (BLOCK_N, BLOCK_K // group_size), scales_a: (BLOCK_M, BLOCK_K // group_size)
        sb_smem = block_size_n * (block_size_k // g) * 1
        sa_smem = block_size_m * (block_size_k // g) * 1
        loop_smem = (a_smem + b_smem + sb_smem + sa_smem) * max(num_stages - 1, 1)
        # Triton overlaps output buffer with loop data (reuses same SMEM)
        output_smem = block_size_m * block_size_n * 2  # bf16 output via TMA store
        estimated_smem = max(loop_smem, output_smem)
    elif e > 1:
        # INT packed: need packed B + dequantized B for MMA
        b_smem = (block_size_k // e) * block_size_n * b_sizeof
        b_smem += block_size_k * block_size_n * a_sizeof
        estimated_smem = int((a_smem + b_smem) * num_stages * 1.20)
    else:
        # INT unpacked (8-bit): exact formula
        b_smem = block_size_k * block_size_n * b_sizeof
        estimated_smem = (a_smem + b_smem) * max(num_stages - 1, 1)
    return estimated_smem

def gpu_supports_bfloat16_atomicadd():
    #Triton tl.atomic_add doens't support bfloat16 on older GPUs.
    return (gpu_compute_capability() >= 9) and GEMLITE_USE_NATIVE_ATOMIC_BFP16

_NUM_SMS = None
def get_num_SMs(device=None):
    #Lazy + cached to avoid driver pings and import-time CUDA context init.
    #Should be ok as multi-gpu systems tend to have the same devices.
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    return _NUM_SMS

#Only powers of 2
def generate_autotune_lookup_v1(max_m=16384):
    return [min(2 ** int(math.ceil(math.log2(M))), max_m) if (M > 0) else 0 for M in range(max_m + 1)]

#Powers of 2 but also (power of 2 + next power of 2) / divisor, 
def generate_autotune_lookup_v2(max_m=16384, min_split=32, divisors=[2, 4], mode='next', include_vllm_config=False):
    lookup = [0] * (max_m + 1)
    autotune_vals = set()

    i = 0
    while (val := 2 ** i) <= max_m:
        autotune_vals.add(val)
        next_val = 2 ** (i + 1)
        if val >= min_split and next_val <= max_m:
            for d in divisors:
                interpolated = (val + next_val) // d
                autotune_vals.add(interpolated)
        i += 1

    if(include_vllm_config):
        autotune_vals.update([1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 
                             72, 80, 88, 96, 104, 112, 120, 128, 
                             136, 144, 152, 160, 168, 176, 184, 192,
                             200, 208, 216, 224, 232, 240, 248, 256, 384, 512])
    
    sorted_vals = sorted(autotune_vals)

    for m in range(max_m + 1):
        if(mode == 'next'):
            lookup[m] = min((x for x in sorted_vals if x >= m), default=None) #Next-value
        elif(mode == 'closest'):
            lookup[m] = min(sorted_vals, key=lambda x: (abs(x - m), x < m)) #Closest-Value
        else:
            raise Exception('Invalid mode.')
    return lookup

M_MAXVAL  = 4096 #1024, 4096, 16384
M_MAPPING = generate_autotune_lookup_v2(M_MAXVAL, mode='next')
def get_closest_m(M):
    return M_MAPPING[M] if M <= M_MAXVAL else M_MAXVAL
    
def get_gpu_shared_memory():
    return driver.active.utils.get_device_properties(0).get("max_shared_mem", 0)

###################################################################################
#Cached results to avoid runtime driver calls
IS_HIP = is_hip()