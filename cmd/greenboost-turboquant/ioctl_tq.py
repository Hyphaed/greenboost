"""
GreenBoost TurboQuant — IOCTL interface for GB_IOCTL_SET_TURBOQUANT.

Mirrors the gb_turboquant_req struct from greenboost_ioctl.h:
    struct gb_turboquant_req {
        uint32_t enabled;
        uint32_t bits;
        uint32_t head_dim;
        uint32_t seed;
    };

IOCTL encoding: _IOW('G', 10, struct gb_turboquant_req)
    _IOW: direction=WRITE(1), magic='G'=0x47, cmd=10, size=sizeof(struct)=16
    ioctl nr = (1 << 30) | (size << 16) | (magic << 8) | cmd
             = (1 << 30) | (16 << 16) | (0x47 << 8) | 10
             = 0x40104747 + adjusted for _IOW
"""

import fcntl
import struct
import os

# IOCTL magic matches greenboost_ioctl.h
GB_IOCTL_MAGIC = ord('G')

# GB_TURBOQUANT_REQ struct: 4 × uint32 = 16 bytes
_TQ_REQ_FMT  = 'IIII'
_TQ_REQ_SIZE = struct.calcsize(_TQ_REQ_FMT)  # 16 bytes

# _IOW(magic, nr, size): direction=WRITE=1, size in bits 29:16, magic in 15:8, nr in 7:0
_IOC_WRITE = 1
_IOC_NONE  = 0

def _ioc(direction, magic, nr, size):
    return ((direction << 30) | (size << 16) | (magic << 8) | nr)

def _iow(magic, nr, size):
    return _ioc(_IOC_WRITE, magic, nr, size)

# GB_IOCTL_SET_TURBOQUANT = _IOW('G', 10, struct gb_turboquant_req)
GB_IOCTL_SET_TURBOQUANT = _iow(GB_IOCTL_MAGIC, 10, _TQ_REQ_SIZE)

# Alias for import by main.py
_iowr_turboquant = GB_IOCTL_SET_TURBOQUANT


def set_turboquant(fd, enabled, bits, head_dim=128, seed=42):
    """
    Send GB_IOCTL_SET_TURBOQUANT to /dev/greenboost.

    Args:
        fd:       open file descriptor for /dev/greenboost
        enabled:  1 = enable TurboQuant, 0 = disable
        bits:     quantization bits: 2, 3, or 4 (ignored when enabled=0)
        head_dim: attention head dimension (default 128)
        seed:     rotation matrix seed (default 42)

    Raises:
        OSError on ioctl failure.
    """
    req = struct.pack(_TQ_REQ_FMT,
                      int(enabled),
                      int(bits),
                      int(head_dim),
                      int(seed))
    fcntl.ioctl(fd, GB_IOCTL_SET_TURBOQUANT, req)
