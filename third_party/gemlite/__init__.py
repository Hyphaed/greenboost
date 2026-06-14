__version__ = "0.6.0"
__author__  = 'Dr. Hicham Badri'
__credits__ = 'Mobius Labs GmbH'

from .core import (
    GemLiteLinearTriton,
    GemLiteLinear,
    DType,
    GEMLITE_ACC_DTYPE,
    set_autotune_setting,
    set_packing_bitwidth,
    set_acc_dtype,
    set_autotune,
    set_kernel_caching,
    enable_tma,
    enable_warp_specialize,
    set_native_atomic_bfp16,
    set_ptx_fp4_pack,
    enable_cudagraph_autotune,
    set_fast_nvfp4,
    forward_functional,
    auto_detect_ptx_fp4_pack,
)

from . import helper

load_config  = GemLiteLinear.load_config
cache_config = GemLiteLinear.cache_config
reset_config = GemLiteLinear.reset_config

# Auto-detect hardware FP4 packing support
auto_detect_ptx_fp4_pack()
