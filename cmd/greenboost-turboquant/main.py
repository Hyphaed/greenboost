#!/usr/bin/env python3
"""
GreenBoost TurboQuant daemon — auto-selects KV cache compression bit width.

Polls the Ollama API every POLL_INTERVAL seconds. When a model is loaded,
estimates the KV cache size and selects the optimal TurboQuant compression
level (turbo2/3/4) to fit the KV cache in T2 DDR while maximizing quality.

Communication paths:
  1. IOCTL: GB_IOCTL_SET_TURBOQUANT → /dev/greenboost (preferred, bare metal)
  2. Conf file: /run/greenboost/turboquant.conf → shim reads every 64 allocs

Systemd integration: sends READY=1 via sd_notify on startup.
"""

import os
import re
import sys
import time
import logging
import socket

from .detector import query_ollama_ps
from .policy import select_bits, estimate_kv_mb, COMPRESSION_RATIOS
from .ioctl_tq import set_turboquant, GB_IOCTL_SET_TURBOQUANT, _iowr_turboquant  # noqa: F401

CONF_PATH     = '/run/greenboost/turboquant.conf'
DEV_PATH      = '/dev/greenboost'
POLL_INTERVAL = 5  # seconds


def write_conf(bits, ratio, head_dim=128, seed=42):
    """
    Atomically write TurboQuant config to /run/greenboost/turboquant.conf.

    The shim reads this file every GB_KV_REFRESH_INTERVAL (64) allocs.
    Atomic write via tmp + rename prevents partial reads.
    """
    os.makedirs('/run/greenboost', exist_ok=True)
    tmp_path = CONF_PATH + '.tmp'
    with open(tmp_path, 'w') as f:
        f.write(f'enabled={1 if bits else 0}\n')
        f.write(f'bits={bits if bits else 0}\n')
        f.write(f'head_dim={head_dim}\n')
        f.write(f'ratio={ratio:.2f}\n')
        f.write(f'seed={seed}\n')
    os.replace(tmp_path, CONF_PATH)


def _read_sysfs_mem():
    """
    Parse KV T1 reserve and T2 available from sysfs status.

    Returns:
        (kv_reserve_mb, t2_available_mb) tuple of ints.
        Falls back to safe defaults if sysfs is unavailable.
    """
    try:
        with open('/sys/class/greenboost/greenboost/status') as f:
            status = f.read()
        kv_rsv   = int(re.search(r'KV cache T1 reserve\s*:\s*(\d+)', status).group(1))
        t2_avail = int(re.search(r'T2 available\s*:\s*(\d+)', status).group(1))
        return kv_rsv, t2_avail
    except Exception:
        # Safe defaults: 2048 MB KV reserve, 32 GB T2 available
        return 2048, 32768


def _sd_notify_ready():
    """Send READY=1 to systemd via the sd_notify socket if available."""
    notify_socket = os.environ.get('NOTIFY_SOCKET')
    if not notify_socket:
        return
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.sendto(b'READY=1', notify_socket)
        sock.close()
    except Exception:
        pass


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[greenboost-turboquant] %(levelname)s %(message)s',
    )
    log = logging.getLogger('greenboost-turboquant')

    _sd_notify_ready()

    gb_fd = -1
    try:
        gb_fd = os.open(DEV_PATH, os.O_RDWR | os.O_CLOEXEC)
        log.info(f'Opened {DEV_PATH} — IOCTL path active')
    except OSError:
        log.warning(f'{DEV_PATH} not available — conf-file path only (no kernel module?)')

    last_model = None
    last_bits  = -1  # -1 = uninitialized, force first write

    log.info(f'Started — polling Ollama every {POLL_INTERVAL}s')

    while True:
        try:
            model_info = query_ollama_ps()

            if model_info and model_info['name']:
                model_name = model_info['name']
                changed = (model_name != last_model)

                if changed:
                    kv_mb = estimate_kv_mb(
                        model_info['size_gb'],
                        model_info['ctx_len'],
                        model_info['kv_type'],
                    )
                    t1_reserve_mb, t2_avail_mb = _read_sysfs_mem()
                    bits = select_bits(kv_mb, t1_reserve_mb, t2_avail_mb)
                    ratio = COMPRESSION_RATIOS.get(bits, 1.0)

                    if bits != last_bits:
                        if bits == 0:
                            log.info(
                                f'TurboQuant disabled for {model_name} '
                                f'(KV ~{kv_mb:.0f} MB fits in T1 reserve {t1_reserve_mb} MB)'
                            )
                        else:
                            compressed_mb = kv_mb / ratio
                            log.info(
                                f'TurboQuant auto-enabled: turbo{bits} for {model_name} '
                                f'(KV ~{kv_mb:.0f} MB → ~{compressed_mb:.0f} MB, '
                                f'ratio {ratio}×, T2 avail {t2_avail_mb} MB)'
                            )
                            if kv_mb / ratio > t2_avail_mb:
                                log.warning(
                                    f'KV compressed size ({compressed_mb:.0f} MB) still exceeds '
                                    f'T2 available ({t2_avail_mb} MB) — using turbo2 best effort'
                                )

                        if gb_fd >= 0:
                            try:
                                set_turboquant(gb_fd,
                                               enabled=int(bits > 0),
                                               bits=bits if bits > 0 else 0)
                            except OSError as e:
                                log.warning(f'IOCTL set_turboquant failed: {e}')

                        write_conf(bits, ratio)
                        last_bits = bits

                    last_model = model_name

            elif not model_info and last_model is not None:
                # Model was unloaded
                log.info(f'Model unloaded — disabling TurboQuant')

                if gb_fd >= 0:
                    try:
                        set_turboquant(gb_fd, enabled=0, bits=0)
                    except OSError as e:
                        log.warning(f'IOCTL disable failed: {e}')

                try:
                    os.unlink(CONF_PATH)
                except OSError:
                    pass

                last_model = None
                last_bits  = -1

        except Exception as e:
            log.debug(f'Poll iteration error: {e}')

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
