# SPDX-License-Identifier: Apache-2.0
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2025

import torch
import gc
import time
from tqdm import tqdm
from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE
from gemlite.triton_kernels.utils import M_MAPPING
from gemlite.triton_kernels.config import BLOCK_QUANT_SIZE
from gemlite.quant_utils import WeightQuantizerMXFP
from .triton_kernels.utils import IS_HIP

if IS_HIP:
    default_fp8 = torch.float8_e4m3fnuz #AMD
    #default_fp8 = torch.float8_e5m2fnuz #AMD
    #default_fp8 = torch.float8_e5m2 #AMD - fp16 emulated
    default_post_scale = True
else:
    default_fp8 = torch.float8_e4m3fn #Nvidia 
    #default_fp8 = torch.float8_e5m2 #Nvidia
    default_post_scale = False

#################################################################################################
#Clean-up
def cleanup_linear(linear_layer, del_orig=True):
    if del_orig:
        for attr in ['weight', 'bias', 'weight_scale', 'W_q', 'meta']:
            val = getattr(linear_layer, attr, None)
            if val is not None and hasattr(val, '__len__') and len(val) > 0:
                setattr(linear_layer, attr, None)
    torch.cuda.empty_cache()

#Replaces all linear layers with the corresponding processor
def patch_model(model, device, processor, skip_modules=['lm_head', 'vision', 'visual'], group_size=64):
    """
    Helper function to quantize the whole model from cpu device. 
    The group_size parameter is only used in HQQ for W_nbits <=4, all the other configs do not use this parameter.
    """
    if(hasattr(processor, "from_hqqlinear")):
        try:
            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        except ImportError:
            print("This processor requires the `hqq` package. Install it via `pip install hqq`.")
    else:
        class _NoHQQ: pass
        HQQLinear = _NoHQQ

    #Name modules
    for name, module in model.named_modules():
        module.name = name

    #Patching fct
    def _patching_fct(layer, device, skip_modules):
        layer = layer.to(device=device, non_blocking=True)
        if(any(s in layer.name for s in skip_modules)):
            return layer
        else:
            if(hasattr(processor, "from_hqqlinear")):
                if(isinstance(layer, torch.nn.Linear)):
                    W_nbits = processor.W_nbits
                    quant_config = BaseQuantizeConfig(nbits=W_nbits, group_size=group_size if W_nbits <=4 else None)
                    layer = HQQLinear(layer, quant_config=quant_config, compute_dtype=layer.weight.dtype, device=device)
                    layer = processor.from_hqqlinear(layer)
            else:
                layer = processor.from_linear(layer)

        return layer

    #Replaces linear layers.  Shared modules (the same Linear object registered
    #under several parents, e.g. LTX-2's patchify_proj living on both the
    #transformer and its args preprocessor) must map to ONE replacement:
    #quantizing strips the original's weights, so a second parent that kept the
    #original would call F.linear(weight=None) at runtime.
    _replacement_by_id = {}
    def _patch_linearlayers(model, fct, device, skip_modules):
        for name, layer in model.named_children():
            if isinstance(layer, (torch.nn.Linear, HQQLinear)):
                key = id(layer)
                if key not in _replacement_by_id:
                    _replacement_by_id[key] = fct(layer, device, skip_modules)
                setattr(model, name, _replacement_by_id[key])
            else:
                _patch_linearlayers(layer, fct, device, skip_modules)

    #Apply patch
    _patch_linearlayers(model, _patching_fct, device, skip_modules)

    #Clean-up
    model = model.to(device=device)
    torch.cuda.empty_cache()
    gc.collect()

    return model

#16-bit activations / 8-bit weigths
class A16W8: #INT8/FP8 weight-only channel-wise
    def __init__(self, device='cuda:0', dtype=None, fp8=None, fp32_scale=True, post_scale=False):
        self.device = device
        self.dtype = dtype
        self.fp8 = fp8
        self.fp32_scale = fp32_scale
        self.post_scale = post_scale

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]

        if(scales is None):
            #Quantize
            if(self.fp8): #FP8
                w_dtype, min_val, max_val = self.fp8, torch.finfo(self.fp8).min, torch.finfo(self.fp8).max
            else: #INT8
                w_dtype, min_val, max_val = torch.int8, torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max

            dtype = weight.dtype if(self.dtype is None) else self.dtype
            
            assert dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ], f"Invalid weight dtype, should be floating point, got {dtype}"

            gemlite_dtype = TORCH_TO_DTYPE[dtype]
            data_ptr = weight.data_ptr()
            weight = weight.to(dtype=torch.float32, copy=False, device=self.device)
            scales = weight.abs().amax(axis=1, keepdim=True) / max_val
            scales.clamp_(min=1e-6)
            W_q = (weight / scales).clamp_(min=min_val, max=max_val)
            if(w_dtype.is_floating_point):
                W_q = W_q.to(w_dtype)
            else:
                W_q = W_q.round_().to(w_dtype)
            if(data_ptr != weight.data_ptr()):
                del weight
                torch.cuda.empty_cache()
        else:
            #Pre-Quantized
            assert weight.itemsize == 1, f"Invalid weight.dtype, should be 8-bit (INT8 or FP8), got {weight.dtype}"
            if(self.dtype is None):
                dtype = scales.dtype if scales.dtype in [torch.float16, torch.bfloat16] else torch.float16
            else:
                dtype = self.dtype
            W_q = weight.to(device=self.device)
            scales = scales.to(device=self.device)
            gemlite_dtype = TORCH_TO_DTYPE[dtype]

        #scales = scales.to(dtype=torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(8, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=gemlite_dtype, 
                        output_dtype=gemlite_dtype, 
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias)

        if(self.post_scale):
            #Post-scale
            gemlite_linear.W_group_mode       = 0
            gemlite_linear.channel_scale_mode = 1
        else:
            #Pre-scaling
            gemlite_linear.W_group_mode       = 2
            gemlite_linear.channel_scale_mode = 0

        return gemlite_linear

    def from_linear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(weight=linear_layer.weight, bias=linear_layer.bias)
        cleanup_linear(linear_layer, del_orig)
        return out_layer

#FP16 activations / Wn packed weights
class A16Wn:  # 8/4/2-bit weight-only as grouped "INT" / 8/4-bit as MXFP type
    def __init__(
        self,
        device="cuda:0",
        dtype=None,
        packing_bitwidth=None,
        post_scale=default_post_scale,
    ):
        self.post_scale = post_scale
        self.device = device
        self.dtype = dtype
        self.packing_bitwidth = packing_bitwidth
        self.quantizer_mx = None
        self.mx_fp8_dtype = default_fp8

    def from_packed_weights(self, W_q, scales, zeros, W_nbits, group_size, bias=None, quant_type="INT"):
        """Load pre-packed weights (already nibble/byte-packed)."""
        return self.from_weights_(W_q, scales, zeros, W_nbits, group_size, bias, quant_type, packed=True)

    def from_weights(self, W_q, scales, zeros, W_nbits, group_size, bias=None, quant_type = "INT"):
        return self.from_weights_(W_q, scales, zeros, W_nbits, group_size, bias, quant_type)

    def from_weights_(self, W_q, scales, zeros, W_nbits, group_size, bias=None, quant_type = "INT", packed=False):
        assert quant_type in ["INT", "MXFP"], f"Invalid quant_type. Got {quant_type}, valid values are INT, MXFP."

        if(quant_type == "MXFP"):
            assert W_nbits in [8, 4], "Unsupported W_nbit for MXFP quant_dtype."
            assert group_size == 32, "group_size should 32 for MXFP."

        if(isinstance(W_q, torch.nn.Parameter)):
            W_q = W_q.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if(quant_type == "INT"):
            dtype = scales.dtype if(self.dtype is None) else self.dtype
            scales = scales.to(dtype)

            assert scales.dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ], "Invalid scales.dtype, should floating point."
            
            gemlite_dtype = TORCH_TO_DTYPE[dtype]
            scales = scales.to(dtype)
            zeros = zeros.to(dtype)

        elif(quant_type == "MXFP"):
            if(W_nbits == 8):
                assert W_q.dtype in [self.mx_fp8_dtype], f"Unsupported dtype of W_q. got {W_q.dtype}"
            if(W_nbits == 4):
                assert W_q.dtype in [torch.uint8], f"Unsupported dtype of W_q. got {W_q.dtype}"

            dtype = torch.float16 if (self.dtype is None) else self.dtype
            if(dtype == torch.float16):
                gemlite_dtype = DType.MXFP16
            elif(dtype == torch.bfloat16):
                gemlite_dtype = DType.MXBF16
            else:
                raise Exception(f"Unsupported dtype for MXFP. Got {dtype}, supported [torch.float16, torch.bfloat16]")
            self.post_scale = False

            N, K = W_q.shape
            W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)#.float()

        in_features, out_features = W_q.shape[::-1] 

        W_q = W_q.to(self.device)
        scales = scales.to(device=self.device) if (scales is not None) else None
        zeros = zeros.to(device=self.device) if (zeros is not None) else None
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(W_nbits, 
                        group_size=group_size,  
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=gemlite_dtype, 
                        output_dtype=gemlite_dtype, 
                        scaled_activations=False,
                        )

        gemlite_linear.pack(
            W_q, scales, zeros, bias=bias, packing_bitwidth=self.packing_bitwidth, packed=packed,
        )

        if(group_size == in_features and quant_type == "INT"):
            if(self.post_scale):
                gemlite_linear.W_group_mode = 1
                gemlite_linear.channel_scale_mode = 1
            else:
                gemlite_linear.W_group_mode = 3
                gemlite_linear.channel_scale_mode = 0

        return gemlite_linear

    def from_hqqlinear(self, hqq_layer, del_orig=True):
        assert hqq_layer.meta['axis'] == 1, 'Only axis==1 is supported.'

        self.device = hqq_layer.W_q.device

        W_nbits    = hqq_layer.meta['nbits']
        group_size = hqq_layer.meta["group_size"]
        if(group_size is None):
            group_size = hqq_layer.in_features

        #Expects uint8 for Wn quantization!
        W_q    = hqq_layer.unpack(dtype=torch.uint8).view(hqq_layer.meta['shape'])
        scales = hqq_layer.meta['scale'].clone()
        zeros  = hqq_layer.meta['zero'].clone()
        bias   = hqq_layer.bias.clone() if (hqq_layer.bias is not None) else None  

        cleanup_linear(hqq_layer, del_orig)

        out_layer = self.from_weights_(
            W_q=W_q,
            scales=scales,
            zeros=zeros,
            W_nbits=W_nbits,
            group_size=group_size,
            bias=bias,
            quant_type="INT",
        )
        
        #Clean-up
        del W_q
        torch.cuda.empty_cache()

        return out_layer

    def mxfp_from_linear(self, linear_layer, W_nbits, del_orig=True):
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        group_size = 32
        N, K = W.shape
        if(W_nbits == 8):
            W_q, scales = self.quantizer_mx.quantize_mxfp8(W, index=True, mx_fp8_dtype=self.mx_fp8_dtype)
        if(W_nbits == 4):
            W_q, scales = self.quantizer_mx.quantize_mxfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)
        
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights_(
            W_q=W_q,
            scales=scales,
            zeros=None,
            W_nbits=W_nbits,
            group_size=group_size,
            bias=bias,
            quant_type="MXFP",
        )

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

#Alias
class A16W8_INT8(A16W8):
    def __init__(self, device='cuda:0', dtype=None, fp32_scale=True, post_scale=False):
        super().__init__(device=device, dtype=dtype, fp8=None, fp32_scale=fp32_scale, post_scale=post_scale)

class A16W8_FP8(A16W8):
    def __init__(self, device='cuda:0', dtype=None, fp8=default_fp8, fp32_scale=True, post_scale=False):
        super().__init__(device=device, dtype=dtype, fp8=fp8, fp32_scale=fp32_scale, post_scale=post_scale)

class A16Wn_HQQ_INT(A16Wn):
    def __init__(self, device='cuda:0', dtype=None, W_nbits=None):
        super().__init__(device=device, dtype=dtype)
        self.W_nbits = W_nbits

    def from_packed_weights(self, W_q_packed, scales, zeros, bias=None):
        """Load pre-packed weights (already nibble/byte-packed)."""
        group_size = W_q_packed.numel() * (self.W_nbits // 4) // scales.numel()  # adjust for packing
        return super().from_packed_weights(
            W_q_packed, scales, zeros, self.W_nbits, group_size, bias=bias,
        )

    def from_weights(self, W_q, scales, zeros, bias=None):
        group_size = W_q.numel() // scales.numel()
        return super().from_weights(
            W_q=W_q,
            scales=scales,
            zeros=zeros,
            W_nbits=self.W_nbits,
            group_size=group_size,
            bias=bias,
            quant_type="INT",
        )

class A16W8_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=8)

class A16W4_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=4)

class A16W2_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=2)

class A16W1_HQQ_INT(A16Wn_HQQ_INT):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=1)

class A16Wn_MXFP(A16Wn):
    def __init__(self, device='cuda:0', dtype=None, W_nbits=None):
        super().__init__(device=device, dtype=dtype)
        self.W_nbits = W_nbits

    def from_packed_weights(self, W_q_packed, scales, bias=None):
        """Load pre-packed MXFP weights (nibble/byte-packed)."""
        group_size = W_q_packed.numel() * (self.W_nbits // 4) // scales.numel()
        return super().from_packed_weights(
            W_q_packed, scales, zeros=None, W_nbits=self.W_nbits,
            group_size=group_size, bias=bias, quant_type="MXFP",
        )

    def from_weights(self, W_q, scales, bias=None):
        group_size = W_q.numel() // scales.numel()
        return super().from_weights(
            W_q=W_q,
            scales=scales,
            zeros=None,
            W_nbits=self.W_nbits,
            group_size=group_size,
            bias=bias,
            quant_type="MXFP",
        )

    def from_linear(self, linear_layer, del_orig=True):
        return super().mxfp_from_linear(
            linear_layer=linear_layer, W_nbits=self.W_nbits, del_orig=del_orig
        )

class A16W8_MXFP(A16Wn_MXFP):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=8)

class A16W4_MXFP(A16Wn_MXFP):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__(device=device, dtype=dtype, W_nbits=4)


class A16W4_NVFP:
    """Weight-only NVFP4 quantization (FP16/BF16 activations, 4-bit NVFP weights).
    Uses group_size=16, FP8 e4m3fn scales, and a global meta_scale."""
    def __init__(self, device='cuda:0', dtype=None):
        self.device = device
        self.dtype = dtype
        self.quantizer_mx = None
        self.W_nbits = 4
        self.group_size = 16
        self.input_dtype = DType.NVFP4

    def from_packed_weights(self, weight_packed, bias=None, scales=None, meta_scale=None):
        """Load pre-packed NVFP4 weights (nibble-packed [N, K//2] uint8).
        Scales: fp8_e4m3fn or uint8 (same bit pattern) [N, K//16].
        meta_scale: global scale factor (fp32)."""
        return self.from_weights(weight_packed, bias=bias, scales=scales, meta_scale=meta_scale, packed=True)

    def from_weights(self, weight, bias=None, scales=None, meta_scale=None, packed=False):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]
        if packed:
            in_features *= 2

        assert scales is not None, "Scales parameter cannot be None. Use from_linear() call to pre-quantize the weights."
        assert weight.dtype in [torch.uint8], f"Invalid weight.dtype, should be uint8, got {weight.dtype}."
        assert scales.dtype in [torch.float8_e4m3fn, torch.uint8], f"Invalid scales.dtype, should be float8_e4m3fn, got {scales.dtype}."
        assert self.group_size == 16, f"Only group_size=16 is supported for NVFP4, got {self.group_size}"
        assert self.dtype is not None, f"Input dtype should be either torch.float16 or torch.bfloat16, not None."

        dtype = self.dtype
        gemlite_dtype = TORCH_TO_DTYPE[dtype]

        W_q = weight.to(device=self.device)
        scales = scales.to(device=self.device)
        if scales.dtype == torch.uint8:
            scales = scales.view(torch.float8_e4m3fn)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(self.W_nbits,
                        group_size=self.group_size,
                        in_features=in_features,
                        out_features=out_features,
                        input_dtype=self.input_dtype,
                        output_dtype=gemlite_dtype,
                        scaled_activations=False,
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias, packed=packed)

        # meta_scale for NVFP4: kernel multiplies acc by meta_scale_norm.
        # Dequant: W ~ W_q * scale / meta_scales, dot_scaled gives W_q * scale,
        # so we need to multiply by 1/meta_scales.
        if meta_scale is not None:
            gemlite_linear.meta_scale = torch.nn.Parameter(
                (1.0 / meta_scale).to(dtype=torch.float32, device=gemlite_linear.W_q.device).reshape(()),
                requires_grad=False,
            )
        return gemlite_linear

    def from_linear(self, linear_layer, del_orig=True):
        if self.dtype is None:
            self.dtype = linear_layer.weight.dtype
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None
        N, K = W.shape
        W_q, scales, _meta_scale = self.quantizer_mx.quantize_nvfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // self.group_size)

        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights(weight=W_q, scales=scales, bias=bias, meta_scale=_meta_scale)

        del W_q
        torch.cuda.empty_cache()
        return out_layer

#################################################################################################
#8-bit dynamic activations / 8-bit weights
#################################################################################################
class A8W8_dynamic:
    def __init__(self, device='cuda:0', dtype=None, fp8=False, fp32_scale=True, block_quant=False):
        self.device = device
        self.dtype = dtype
        self.fp8 = fp8
        self.fp32_scale = fp32_scale
        self.block_quant = block_quant

    def _from_weights_block_quant(self, weight, bias=None, scales=None):
        """BxB block quantization for weights + per-block (row x B) activations, B=BLOCK_QUANT_SIZE.

        If ``scales`` is None, ``weight`` is quantized in-place (expected to be a
        floating-point tensor) using per-block amax. If ``scales`` is provided,
        ``weight`` must already be pre-quantized (FP8/INT8) and ``scales`` must
        be shape ``[N//B, K//B]`` in fp32 — no re-quantization is performed.
        """
        if(self.fp8):
            w_dtype, input_dtype = self.fp8, TORCH_TO_DTYPE[self.fp8]
            min_val, max_val = torch.finfo(self.fp8).min, torch.finfo(self.fp8).max
        else:
            w_dtype, input_dtype = torch.int8, DType.INT8
            min_val, max_val = torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max

        in_features, out_features = weight.shape[::-1]
        B = BLOCK_QUANT_SIZE

        if scales is not None:
            # Pre-quantized path: weight is already w_dtype, scales already [N//B, K//B].
            assert weight.dtype == w_dtype, (
                f"Pre-quantized weight must be {w_dtype}, got {weight.dtype}"
            )
            expected_scale_shape = (
                (out_features + B - 1) // B,
                (in_features + B - 1) // B,
            )
            assert tuple(scales.shape) == expected_scale_shape, (
                f"Pre-quantized scales must have shape {expected_scale_shape}, "
                f"got {tuple(scales.shape)}"
            )
            if not weight.is_contiguous():
                weight = weight.contiguous()
            W_q = weight.to(device=self.device)
            scales = scales.to(device=self.device, dtype=torch.float32)
            dtype = self.dtype if (self.dtype is not None) else torch.bfloat16
            gemlite_dtype = TORCH_TO_DTYPE[dtype]
        else:
            # Self-quantization path: weight is floating-point, we compute scales.
            dtype = weight.dtype if(self.dtype is None) else self.dtype
            assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid weight dtype, should be floating point."
            gemlite_dtype = TORCH_TO_DTYPE[dtype]

            in_pad  = ((in_features  + B - 1) // B) * B
            out_pad = ((out_features + B - 1) // B) * B
            pad_k, pad_n = in_pad - in_features, out_pad - out_features

            weight = weight.to(dtype=torch.float32, device=self.device)
            if pad_k or pad_n:
                weight_padded = torch.nn.functional.pad(weight, (0, pad_k, 0, pad_n))
            else:
                weight_padded = weight
            # Reshape to [N//B, B, K//B, B] and compute per-block amax on the padded view
            w = weight_padded.view(out_pad // B, B, in_pad // B, B)
            amax = w.abs().amax(dim=(1, 3))  # [N//B, K//B]
            scales = (amax / max_val).clamp_(min=1e-6)  # [N//B, K//B] fp32
            W_q = (w / scales[:, None, :, None]).clamp_(min=min_val, max=max_val)
            if(w_dtype.is_floating_point):
                W_q = W_q.to(w_dtype)
            else:
                W_q = W_q.round_().to(w_dtype)
            W_q = W_q.view(out_pad, in_pad)
            if pad_k or pad_n:
                # Drop the zero-padded rows/cols; keep the layer at its original shape.
                W_q = W_q[:out_features, :in_features].contiguous()

        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(
            8,
            group_size=in_features,  # dummy: block-quant doesn't use group_size
            in_features=in_features,
            out_features=out_features,
            input_dtype=input_dtype,
            output_dtype=gemlite_dtype,
            acc_dtype=DType.FP32,  # fp32 scales => accumulate in fp32
            scaled_activations=True,
        )

        # Follow existing pack() contract: it will scale.view((out_features, -1)).t() which gives [?, out_features].
        # For block quant, we want scales stored as [K//B, N//B] fp32 so that stride_meta_g = stride(0) and stride_meta_n = stride(1).
        # Pass a placeholder 1D scale to pack() (so it sets meta_is_channelwise=True path), then overwrite.
        placeholder = torch.ones((out_features, 1), dtype=torch.float32, device=self.device)
        gemlite_linear.pack(W_q, scales=placeholder, zeros=None, bias=bias)

        # Replace scales buffer with block-quant layout [K//B, N//B] fp32 — shape is
        # ceil(K/B) x ceil(N/B) so kernel lookups for the partial edge blocks remain in bounds.
        block_scales = scales.t().contiguous()  # [K//B, N//B]
        gemlite_linear.scales = torch.nn.Parameter(block_scales, requires_grad=False)
        gemlite_linear.meta_dtype = DType.FP32

        # channel_scale_mode=5: INT/FP8 in-kernel block-scale multiply, no W_group dequant on B
        gemlite_linear.W_group_mode = 0
        gemlite_linear.channel_scale_mode = 5

        # Refresh metadata tensor to reflect updated modes/meta_dtype
        gemlite_linear.metadata = torch.nn.Parameter(
            torch.tensor(gemlite_linear.get_meta_args(), device=self.device, dtype=torch.int32),
            requires_grad=False,
        )
        return gemlite_linear

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if self.block_quant:
            return self._from_weights_block_quant(weight, bias=bias, scales=scales)

        if(self.fp8): #FP8
            w_dtype, input_dtype, min_val, max_val = self.fp8, TORCH_TO_DTYPE[self.fp8], torch.finfo(self.fp8).min, torch.finfo(self.fp8).max
        else: #INT8
            w_dtype, input_dtype, min_val, max_val = torch.int8, DType.INT8, torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max

        in_features, out_features = weight.shape[::-1]

        if(scales is None):
            #Quantize
            dtype = weight.dtype if(self.dtype is None) else self.dtype

            assert dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ], "Invalid weight dtype, should be floating point."

            gemlite_dtype = TORCH_TO_DTYPE[dtype]
            data_ptr = weight.data_ptr()
            weight = weight.to(dtype=torch.float32, copy=False, device=self.device)
            scales = weight.abs().amax(axis=1, keepdim=True) / max_val
            scales.clamp_(min=1e-6)
            W_q = (weight / scales).clamp_(min=min_val, max=max_val)
            if(w_dtype.is_floating_point):
                W_q = W_q.to(w_dtype)
            else:
                W_q = W_q.round_().to(w_dtype)
            if(data_ptr != weight.data_ptr()):
                del weight
                torch.cuda.empty_cache()
        else:
            #Pre-Quantized
            assert weight.dtype.itemsize == 1, "Invalid weight.dtype, should be 8-bit."
            if(self.dtype is None):
                dtype = scales.dtype if scales.dtype in [torch.float16, torch.bfloat16] else torch.float16
            else:
                dtype = self.dtype
            W_q = weight.to(device=self.device)
            scales = scales.to(device=self.device)
            gemlite_dtype = TORCH_TO_DTYPE[dtype]

        scales = scales.to(torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(8,
                        group_size=in_features,
                        in_features=in_features,
                        out_features=out_features,
                        input_dtype=input_dtype,
                        output_dtype=gemlite_dtype,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias)

        #Post-scaling
        gemlite_linear.W_group_mode       = 0
        gemlite_linear.channel_scale_mode = 3
        return gemlite_linear

    def from_linear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(weight=linear_layer.weight, bias=linear_layer.bias)
        cleanup_linear(linear_layer, del_orig)
        return out_layer

class A8W8_int8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', dtype=None, block_quant=False):
        super().__init__(block_quant=block_quant)
        self.device = device
        self.dtype = dtype
        self.fp8 = False
A8W8_INT8_dynamic = A8W8_int8_dynamic

class A8W8_fp8_dynamic(A8W8_dynamic):
    def __init__(self, device='cuda:0', dtype=None, fp8=default_fp8, block_quant=False):
        super().__init__(block_quant=block_quant)
        self.device = device
        self.dtype = dtype
        self.fp8 = fp8
A8W8_FP8_dynamic = A8W8_fp8_dynamic

#################################################################################################
#FP8 dynamic activations / W4 packed weights
#################################################################################################
class A8Wn_HQQ_INT_dynamic(A16Wn):
    def __init__(
        self,
        device="cuda:0",
        packing_bitwidth=None,
        dtype=None,
        post_scale=default_post_scale,
        fp8=default_fp8,
        fp32_scale=False,
        W_nbits=None,
    ):
        assert W_nbits is not None, "W_nbits argument should be eitehr 8 or 4, not None)."
        super().__init__()
        self.post_scale = post_scale
        self.device = device
        self.dtype = dtype
        self.packing_bitwidth = packing_bitwidth
        self.fp8 = fp8
        self.fp32_scale = fp32_scale
        self.W_nbits = W_nbits

    def from_packed_weights(self, W_q_packed, scales, zeros, bias=None):
        """Load pre-packed weights (already nibble/byte-packed)."""
        group_size = W_q_packed.numel() * (self.W_nbits // 4) // scales.numel()
        return self.from_weights_(W_q_packed, scales, zeros, self.W_nbits, group_size, bias=bias, packed=True)

    def from_weights(self, W_q, scales, zeros, bias=None):
        group_size = W_q.numel() // scales.numel()
        return self.from_weights_(
            W_q=W_q,
            scales=scales,
            zeros=zeros,
            W_nbits=self.W_nbits,
            group_size=group_size,
            bias=bias,
        )

    def from_weights_(self, W_q, scales, zeros, W_nbits, group_size, bias=None, packed=False):
        if(isinstance(W_q, torch.nn.Parameter)):
            W_q = W_q.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        if(self.dtype is None):
            dtype = scales.dtype if scales.dtype in [torch.float16, torch.bfloat16] else torch.float16
        else:
            dtype = self.dtype 

        assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Invalid scales.dtype, should be floating point."
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        input_dtype = TORCH_TO_DTYPE[self.fp8]
            
        W_q = W_q.to(self.device)
        scales = scales.to(device=self.device, dtype=torch.float32 if self.fp32_scale else dtype) if (scales is not None) else None
        zeros = zeros.to(device=self.device, dtype=dtype) if (zeros is not None) else None
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        in_features, out_features = W_q.shape[::-1]

        gemlite_linear = GemLiteLinearTriton(W_nbits, 
                        group_size=group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype, 
                        output_dtype=gemlite_dtype,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(
            W_q,
            scales,
            zeros,
            bias=bias,
            packing_bitwidth=self.packing_bitwidth,
            fma_mode=False,
            packed=packed,
        )

        if(group_size == in_features):
            if(self.post_scale):
                gemlite_linear.W_group_mode = 1
                gemlite_linear.channel_scale_mode = 3
            else:
                gemlite_linear.W_group_mode = 3
                gemlite_linear.channel_scale_mode = 2

        return gemlite_linear

    def from_hqqlinear(self, hqq_layer, del_orig=True):
        assert hqq_layer.meta['axis'] == 1, 'Only axis==1 is supported.'

        self.device = hqq_layer.W_q.device

        W_nbits    = hqq_layer.meta['nbits']
        group_size = hqq_layer.meta["group_size"]
        if(group_size is None):
            group_size = hqq_layer.in_features

        W_q    = hqq_layer.unpack(dtype=torch.uint8).view(hqq_layer.meta['shape']) #Expects uint8 for Wn quantization!
        scales = hqq_layer.meta['scale'].clone()
        zeros  = hqq_layer.meta['zero'].clone()
        bias   = hqq_layer.bias.clone() if (hqq_layer.bias is not None) else None
        
        cleanup_linear(hqq_layer, del_orig)

        out_layer = self.from_weights_(
            W_q=W_q,
            scales=scales,
            zeros=zeros,
            W_nbits=W_nbits,
            group_size=group_size,
            bias=bias,
        )
        
        #Clean-up
        del W_q
        torch.cuda.empty_cache()

        return out_layer

class A8W4_HQQ_INT_dynamic(A8Wn_HQQ_INT_dynamic):
    def __init__(
        self,
        device="cuda:0",
        packing_bitwidth=None,
        dtype=None,
        post_scale=default_post_scale,
        fp8=default_fp8,
        fp32_scale=False,
    ):
        super().__init__(
            device=device,
            packing_bitwidth=packing_bitwidth,
            dtype=dtype,
            post_scale=post_scale,
            fp8=fp8,
            fp32_scale=fp32_scale,
            W_nbits=4,
        )


class A8W2_HQQ_INT_dynamic(A8Wn_HQQ_INT_dynamic):
    def __init__(
        self,
        device="cuda:0",
        packing_bitwidth=None,
        dtype=None,
        post_scale=default_post_scale,
        fp8=default_fp8,
        fp32_scale=False,
    ):
        super().__init__(
            device=device,
            packing_bitwidth=packing_bitwidth,
            dtype=dtype,
            post_scale=post_scale,
            fp8=fp8,
            fp32_scale=fp32_scale,
            W_nbits=2,
        )

#################################################################################################
#MXFP / NVFP dtypes with dynamic quant activations
#################################################################################################
class A8W8_MXFP_dynamic:
    def __init__(self, device='cuda:0', dtype=None, post_scale=True, fp8=default_fp8):
        self.device = device
        self.dtype = dtype
        self.mx_fp8_dtype = fp8
        self.quantizer_mx = None
        self.post_scale = post_scale
        self.W_nbits = 8

    def from_weights(self, weight, bias=None, scales=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]

        assert scales is not None, "Scales parameter cannot be None. Use from_linear() call to pre-quantize the weights."

        #Pre-Quantized
        assert weight.is_floating_point() and weight.itemsize == 1, f"Invalid weight.dtype, should be an MXPF8 dype, got {weight.dtype}."
        assert scales.dtype in [torch.float8_e8m0fnu, torch.uint8], f"Invalid scales.dtype, should be e8m0 / view(uint8), got {scales.dtype}."
        assert self.dtype is not None, f"Input dtype should be either torch.float16 or torch.bfloat16, not None."
        dtype = self.dtype 
        input_dtype = DType.MXFP8
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        group_size = 32

        W_q = weight.to(device=self.device)
        scales = scales.to(device=self.device) 
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(self.W_nbits, 
                        group_size=group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype,
                        output_dtype=gemlite_dtype,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias)

        #If post_scale==False, it will use mxfp8 microscales for the activations, otherwise channelwise post-scaling is used
        gemlite_linear.W_group_mode       = 0
        gemlite_linear.channel_scale_mode = 2 if self.post_scale else 4
        return gemlite_linear


    def from_linear(self, linear_layer, del_orig=True):
        if self.dtype is None:
            self.dtype = linear_layer.weight.dtype
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        group_size = 32
        N, K = W.shape
        if(self.W_nbits == 8):
            W_q, scales = self.quantizer_mx.quantize_mxfp8(W, index=True, mx_fp8_dtype=self.mx_fp8_dtype)
        if(self.W_nbits == 4):
            W_q, scales = self.quantizer_mx.quantize_mxfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)
        
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights(weight=W_q, scales=scales, bias=bias)

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

class A8Wn_MXFP_dynamic:
    def __init__(self, device='cuda:0', dtype=None, post_scale=True, fp8=default_fp8, W_nbits=None):
        assert W_nbits is not None, "W_nbits argument should be eitehr 8 or 4, not None)."
        self.device = device
        self.dtype = dtype
        self.mx_fp8_dtype = fp8
        self.quantizer_mx = None
        self.post_scale = post_scale
        self.W_nbits = W_nbits

    def from_packed_weights(self, weight_packed, bias=None, scales=None):
        """Load pre-packed MXFP weights (nibble-packed [N, K//2] for 4-bit)."""
        return self.from_weights(weight_packed, bias=bias, scales=scales, packed=True)

    def from_weights(self, weight, bias=None, scales=None, packed=False):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]
        if packed and self.W_nbits == 4:
            in_features *= 2  # packed shape is [N, K//2]

        assert scales is not None, "Scales parameter cannot be None. Use from_linear() call to pre-quantize the weights."

        #Pre-Quantized
        if(self.W_nbits == 8):
            assert weight.is_floating_point() and weight.itemsize == 1, f"Invalid weight.dtype, should be an MXPF8 valid dtype, got {weight.dtype}."
        if(self.W_nbits == 4):
            assert weight.dtype in [torch.uint8], f"Invalid weight.dtype, should be an MXPF8 (FP8 dtype) valid dtype, got {weight.dtype}."
        assert scales.dtype in [torch.float8_e8m0fnu, torch.uint8], f"Invalid scales.dtype, should be e8m0 / view(uint8), got {scales.dtype}."
        assert self.dtype is not None, f"Input dtype should be either torch.float16 or torch.bfloat16, not None."
        dtype = self.dtype 
        input_dtype = DType.MXFP8
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        group_size = 32

        W_q = weight.to(device=self.device)
        scales = scales.to(device=self.device) 
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(self.W_nbits, 
                        group_size=group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype,
                        output_dtype=gemlite_dtype,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias, packed=packed)

        #If post_scale==False, it will use mxfp8 microscales for the activations, otherwise channelwise post-scaling is used
        gemlite_linear.W_group_mode       = 0
        gemlite_linear.channel_scale_mode = 2 if self.post_scale else 4
        return gemlite_linear


    def from_linear(self, linear_layer, del_orig=True):
        if self.dtype is None:
            self.dtype = linear_layer.weight.dtype
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        group_size = 32
        N, K = W.shape
        if(self.W_nbits == 8):
            W_q, scales = self.quantizer_mx.quantize_mxfp8(W, index=True, mx_fp8_dtype=self.mx_fp8_dtype)
        if(self.W_nbits == 4):
            W_q, scales = self.quantizer_mx.quantize_mxfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // group_size)
        
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights(weight=W_q, scales=scales, bias=bias)

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

class A8W8_MXFP_dynamic(A8Wn_MXFP_dynamic):
    def __init__(self, device='cuda:0', dtype=None, post_scale=True, fp8=default_fp8):
        super().__init__(device=device, dtype=dtype, post_scale=post_scale, fp8=fp8, W_nbits=8)

class A8W4_MXFP_dynamic(A8Wn_MXFP_dynamic):
    def __init__(self, device='cuda:0', dtype=None, post_scale=True, fp8=default_fp8):
        super().__init__(device=device, dtype=dtype, post_scale=post_scale, fp8=fp8, W_nbits=4)

class A4W4_MXFP_dynamic:
    def __init__(self, device='cuda:0', dtype=None):
        self.device = device
        self.dtype = dtype
        self.quantizer_mx = None
        self.W_nbits = 4
        self.group_size = 32
        self.input_dtype = DType.MXFP4

    def from_packed_weights(self, weight_packed, bias=None, scales=None):
        """Load pre-packed MXFP4 weights (nibble-packed [N, K//2] uint8)."""
        return self.from_weights(weight_packed, bias=bias, scales=scales, packed=True)

    def from_weights(self, weight, bias=None, scales=None, packed=False):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]
        if packed:
            in_features *= 2  # packed shape is [N, K//2]

        assert scales is not None, "Scales parameter cannot be None. Use from_linear() call to pre-quantize the weights."

        #Pre-Quantized
        assert weight.dtype in [torch.uint8], f"Invalid weight.dtype, should be an MXPF8 valid dtype, got {weight.dtype}."
        assert scales.dtype in [torch.float8_e8m0fnu, torch.uint8], f"Invalid scales.dtype, should be e8m0 / view(uint8), got {scales.dtype}."
        assert self.dtype is not None, f"Input dtype should be either torch.float16 or torch.bfloat16, not None."
        assert self.group_size == 32, f"Only group_size=32 is supported for MXFP4, got {self.group_size}"

        dtype = self.dtype 
        gemlite_dtype = TORCH_TO_DTYPE[dtype]

        W_q = weight.to(device=self.device)
        scales = scales.to(device=self.device) 
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(self.W_nbits, 
                        group_size=self.group_size, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=self.input_dtype,
                        output_dtype=gemlite_dtype,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias, packed=packed)
        gemlite_linear.W_group_mode       = 0
        gemlite_linear.channel_scale_mode = 4
        return gemlite_linear


    def from_linear(self, linear_layer, del_orig=True):
        if self.dtype is None:
            self.dtype = linear_layer.weight.dtype
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None 
        N, K = W.shape
        W_q, scales = self.quantizer_mx.quantize_mxfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // self.group_size)
        
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights(weight=W_q, scales=scales, bias=bias)

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

class A4W4_NVFP_dynamic:
    def __init__(self, device='cuda:0', dtype=None):
        self.device = device
        self.dtype = dtype
        self.quantizer_mx = None
        self.W_nbits = 4
        self.group_size = 16
        self.input_dtype = DType.NVFP4

    def from_packed_weights(self, weight_packed, bias=None, scales=None, meta_scale=None, input_scale=None):
        """Load pre-packed NVFP4 weights (nibble-packed [N, K//2] uint8).
        Scales: fp8_e4m3fn or uint8 (same bit pattern) [N, K//16].
        meta_scale: global scale factor (fp32, = 448*6/global_amax = flashinfer gsf).
        input_scale: ModelOpt-style fp32 scalar = global_amax(x)/(448*6). If provided,
        gemlite stores its reciprocal and skips the dynamic amax on activations."""
        return self.from_weights(weight_packed, bias=bias, scales=scales, meta_scale=meta_scale, input_scale=input_scale, packed=True)

    def from_weights(self, weight, bias=None, scales=None, meta_scale=None, input_scale=None, packed=False):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        in_features, out_features = weight.shape[::-1]
        if packed:
            in_features *= 2  # packed shape is [N, K//2]

        assert scales is not None, "Scales parameter cannot be None. Use from_linear() call to pre-quantize the weights."

        #Pre-Quantized
        assert weight.dtype in [torch.uint8], f"Invalid weight.dtype, should be an MXPF8 dtype, got {weight.dtype}."
        assert scales.dtype in [torch.float8_e4m3fn, torch.uint8], f"Invalid scales.dtype, should be float8_e4m3fn, got {scales.dtype}."
        assert self.dtype is not None, f"Input dtype should be either torch.float16 or torch.bfloat16, not None."
        assert self.group_size == 16, f"Only group_size=16 is supported for NVFP4, got {self.group_size}"

        dtype = self.dtype
        gemlite_dtype = TORCH_TO_DTYPE[dtype]

        W_q = weight.to(device=self.device)
        scales = scales.to(device=self.device)
        if scales.dtype == torch.uint8:
            scales = scales.view(torch.float8_e4m3fn)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear = GemLiteLinearTriton(self.W_nbits,
                        group_size=self.group_size,
                        in_features=in_features,
                        out_features=out_features,
                        input_dtype=self.input_dtype,
                        output_dtype=gemlite_dtype,
                        scaled_activations=True,
                        )

        gemlite_linear.pack(W_q, scales, zeros=None, bias=bias, packed=packed)
        gemlite_linear.W_group_mode       = 0
        gemlite_linear.channel_scale_mode = 4
        if meta_scale is not None:
            gemlite_linear.meta_scale = torch.nn.Parameter(
                meta_scale.to(dtype=torch.float32, device=gemlite_linear.W_q.device).reshape(()),
                requires_grad=False,
            )
        if input_scale is not None:
            # ModelOpt input_scale = global_amax/(448*6); gemlite kernel expects 1/input_scale.
            from gemlite.core import GEMLITE_NVFP4_INPUT_SCALES
            _is = input_scale.to(dtype=torch.float32, device=gemlite_linear.W_q.device).reshape(1)
            GEMLITE_NVFP4_INPUT_SCALES[gemlite_linear.W_q.data_ptr()] = (1.0 / _is).contiguous()
        return gemlite_linear


    def from_linear(self, linear_layer, del_orig=True):
        if self.dtype is None:
            self.dtype = linear_layer.weight.dtype
        if(self.quantizer_mx is None):
            self.quantizer_mx = WeightQuantizerMXFP(device=self.device, compute_dtype=linear_layer.weight.dtype)

        W = linear_layer.weight.data
        bias = linear_layer.bias.clone() if (linear_layer.bias is not None) else None
        N, K = W.shape
        W_q, scales, _meta_scale = self.quantizer_mx.quantize_nvfp4(W, index=True)
        W_q, scales = W_q.view([N, K]), scales.view(N, K // self.group_size)
        cleanup_linear(linear_layer, del_orig)

        out_layer = self.from_weights(weight=W_q, scales=scales, bias=bias, meta_scale=_meta_scale)

        #Clean-uo
        del W_q
        torch.cuda.empty_cache()
        return out_layer

#################################################################################################
#BitNet
#################################################################################################
class A16W158_INT:
    def __init__(self, device='cuda:0', dtype=None, fp32_scale=True):
        self.device = device
        self.dtype = dtype
        self.fp32_scale = fp32_scale

    def from_weights(self, weight, weight_scale, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        dtype = weight.dtype if(self.dtype is None) else self.dtype

        assert dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "Invalid weight.dtype, should be floating point."

        weight        = weight.to(dtype=dtype, device=self.device)
        W_q           = (weight + 1).to(torch.uint8)
        bias          = bias if bias is not None else None
        dtype         = weight.dtype
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        out_features  = W_q.shape[0]
        in_features   = W_q.shape[1]
        scales        = torch.ones((out_features, 1), dtype=torch.float32) * weight_scale.item() 

        gemlite_linear = GemLiteLinearTriton(2, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=gemlite_dtype, 
                        output_dtype=gemlite_dtype, 
                        scaled_activations=False,
                        )

        scales = scales.to(dtype=torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear.pack(W_q, scales=scales, zeros=1, bias=bias)
        #post-scale
        gemlite_linear.W_group_mode       = 1 #shift only
        gemlite_linear.channel_scale_mode = 1 #weight-only
        return gemlite_linear

    def from_bitlinear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(
            weight=linear_layer.weight,
            weight_scale=linear_layer.weight_scale,
            bias=linear_layer.bias,
        )
        cleanup_linear(linear_layer, del_orig)
        return out_layer

class A8W158_INT_dynamic:
    def __init__(self, device="cuda:0", dtype=None, fp32_scale=True):
        self.device = device
        self.dtype = dtype
        self.fp32_scale = fp32_scale

    def from_weights(self, weight, weight_scale, bias=None):
        if(isinstance(weight, torch.nn.Parameter)):
            weight = weight.data
        if(isinstance(bias, torch.nn.Parameter)):
            bias = bias.data

        dtype = weight.dtype if(self.dtype is None) else self.dtype

        assert dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "Invalid weight.dtype, should be floating point."

        weight        = weight.to(device=self.device, dtype=dtype)
        W_q           = (weight + 1).to(torch.uint8)
        dtype         = weight.dtype
        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        out_features  = W_q.shape[0]
        in_features   = W_q.shape[1]
        scales        = torch.ones((out_features, 1), dtype=torch.float32, device=self.device) * weight_scale.item()
        input_dtype   = DType.INT8

        gemlite_linear = GemLiteLinearTriton(2, 
                        group_size=in_features, 
                        in_features=in_features, 
                        out_features=out_features, 
                        input_dtype=input_dtype, 
                        output_dtype=gemlite_dtype, 
                        scaled_activations=True,
                        )

        scales = scales.to(dtype=torch.float32 if self.fp32_scale else dtype)
        bias = bias.to(device=self.device, dtype=dtype) if (bias is not None) else None

        gemlite_linear.pack(W_q, scales=scales, zeros=1, bias=bias)

        #post-scale 
        gemlite_linear.W_group_mode       = 1 #shift only
        gemlite_linear.channel_scale_mode = 3 #activations + weight

        return gemlite_linear

    def from_bitlinear(self, linear_layer, del_orig=True):
        out_layer = self.from_weights(
            weight=linear_layer.weight,
            weight_scale=linear_layer.weight_scale,
            bias=linear_layer.bias,
        )
        cleanup_linear(linear_layer, del_orig)
        return out_layer

#################################################################################################
#Warm-up function
#################################################################################################
default_batch_sizes = sorted(list(set(M_MAPPING)))[::-1]
def warmup(
    processor,
    shapes: list[tuple],
    batch_sizes: list[int] = default_batch_sizes,
    group_size: int = 64,
    dtype: torch.dtype = torch.float16,
):
    """
    processor = A8W8_INT8_dynamic()
    warmup(processor, shapes=[(4096, 4096), (4096, 2048)], batch_sizes=[1, 4, 8, ..])
    `group_size` is only used for hqq quantization.
    """

    if(hasattr(processor, "from_hqqlinear")):
        try:
            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        except ImportError:
            print("This processor requires the `hqq` package. Install it via `pip install hqq`.")
    else:
        class _NoHQQ: pass
        HQQLinear = _NoHQQ


    device = torch.device(torch.cuda.current_device())
    
    for shape in shapes:
        out_features, in_features = shape

        linear = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            dtype=dtype,
            device=device,
        )

        if(hasattr(processor, 'from_hqqlinear')):
            W_nbits        = processor.W_nbits
            quant_config   = BaseQuantizeConfig(nbits=W_nbits, group_size=group_size, axis=1)
            linear         = HQQLinear(linear, quant_config=quant_config, compute_dtype=dtype, device=device)
            gemlite_linear = processor.from_hqqlinear(linear)
        else:
            gemlite_linear = processor.from_linear(linear)

        for batch_size in tqdm(batch_sizes):
            _ = gemlite_linear(torch.randn((batch_size, in_features), dtype=dtype, device=device) / 100.)
            torch.cuda.synchronize()

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

