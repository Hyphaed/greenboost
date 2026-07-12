#!/usr/bin/env python3
"""
gb_feeder_diag.py - GreenBoost feeder diagnostic tool

Speaks the net_fabric binary protocol directly to a running greenboost-netd
feeder daemon.  Reads /etc/greenboost/cluster.conf for the feeder address.

Usage:
    python3 gb_feeder_diag.py [t1|t2|t3|compute|all]  [--ip IP] [--port PORT]

Commands:
    t1       - allocate 64 MB on feeder T1 VRAM, verify, free
    t2       - allocate 64 MB on feeder T2 DDR,  verify, free
    t3       - allocate 64 MB on feeder T3 NVMe, verify, free
    compute  - send a kernel exec request and check the feeder response
    all      - run t1 + t2 + t3 + compute in sequence (default)
"""

import hashlib
import hmac
import socket
import struct
import sys
import os
import time
import argparse
import statistics
import concurrent.futures

# ── Protocol constants (must match features/net_fabric.h) ─────────────────────
GB_NET_MAGIC          = 0x47424E46  # "GBNF"
# F-L3-09: seq_num(4) added in proto v3 - header is now 16 bytes.
GB_NET_HDR_FMT        = "<IHHII"    # magic(4) msg_type(2) flags(2) payload_len(4) seq_num(4)
# Derive header size from the format string so a layout change
# in net_fabric.h cannot silently desync this client.
GB_NET_HDR_SIZE       = struct.calcsize(GB_NET_HDR_FMT)
assert GB_NET_HDR_SIZE == 16, f"net_fabric.h header layout drift: {GB_NET_HDR_SIZE}"

GB_MSG_HANDSHAKE_REQ  = 0x01
GB_MSG_HANDSHAKE_RESP = 0x02
GB_MSG_HEARTBEAT      = 0x03
GB_MSG_CUDA_MALLOC    = 0x10
GB_MSG_CUDA_FREE      = 0x11
# Audit F-L5-28: name the EXEC msg type so call sites stop using the 0x23 magic.
GB_MSG_CUDA_EXEC      = 0x23
GB_MSG_MEM_INFO       = 0x31
GB_MSG_FEEDER_STATUS  = 0x34
GB_MSG_RESPONSE       = 0x40

# gb_feeder_status_resp sizes (features/net_fabric.h): v3.0 base struct is 64
# bytes (status..._pad); v3.1 appends 20 bytes of GPU telemetry
# (gpu_temp_c/gpu_power_w/gpu_util_pct/ecc_dbe_count/throttle_reasons/_pad2).
GB_FEEDER_STATUS_V30_SIZE = 64
GB_FEEDER_STATUS_V31_SIZE = 84

GB_NET_FLAG_RESPONSE  = 0x0001

GB_STATUS_OK             = 0
GB_STATUS_ERR_OOM        = 1
GB_STATUS_ERR_INVALID    = 2
GB_STATUS_ERR_CUDA       = 3
GB_STATUS_ERR_THROTTLE   = 7

GB_ALLOC_TIER_AUTO    = 0x00
GB_ALLOC_TIER_T1      = 0x01
GB_ALLOC_TIER_T2      = 0x02
GB_ALLOC_TIER_T3      = 0x04

TIER_NAMES = {0: "T1 GPU VRAM", 1: "T2 DDR", 2: "T3 NVMe/pageable"}
STATUS_NAMES = {
    GB_STATUS_OK: "OK",
    GB_STATUS_ERR_OOM: "OOM",
    GB_STATUS_ERR_INVALID: "INVALID",
    GB_STATUS_ERR_CUDA: "CUDA_ERR",
    GB_STATUS_ERR_THROTTLE: "THROTTLE",
}

# ANSI colours
C_GREEN  = "\033[92m"
C_RED    = "\033[91m"
C_YELLOW = "\033[93m"
C_CYAN   = "\033[96m"
C_DIM    = "\033[2m"
C_BOLD   = "\033[1m"
C_RESET  = "\033[0m"

def _ok(msg):  print(f"  {C_GREEN}✓{C_RESET}  {msg}")
def _fail(msg): print(f"  {C_RED}✗{C_RESET}  {msg}", file=sys.stderr)
def _info(msg): print(f"  {C_CYAN}◈{C_RESET}  {C_DIM}{msg}{C_RESET}")
def _warn(msg): print(f"  {C_YELLOW}⚠{C_RESET}  {msg}")
def _head(msg): print(f"\n{C_BOLD}{msg}{C_RESET}")


# ── Low-level I/O ──────────────────────────────────────────────────────────────

def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("Connection closed by feeder")
        buf += chunk
    return buf

import threading
_seq_local = threading.local()

def _reset_seq():
    _seq_local.send = 0
    _seq_local.recv = 0

def _send_msg(sock, msg_type, payload: bytes = b"", flags: int = 0):
    if not hasattr(_seq_local, "send"):
        _seq_local.send = 0
    hdr = struct.pack("<IHHII", GB_NET_MAGIC, msg_type, flags, len(payload), _seq_local.send)
    _seq_local.send += 1
    sock.sendall(hdr + payload)

def _recv_msg(sock):
    if not hasattr(_seq_local, "recv"):
        _seq_local.recv = 0
    raw = _recvall(sock, GB_NET_HDR_SIZE)
    magic, msg_type, flags, plen, seq = struct.unpack("<IHHII", raw)
    if magic != GB_NET_MAGIC:
        raise ValueError(f"Bad magic: 0x{magic:08X}")
    if seq != _seq_local.recv:
        raise ValueError(f"Seq mismatch: expected {_seq_local.recv}, got {seq}")
    _seq_local.recv += 1
    payload = _recvall(sock, plen) if plen else b""
    return msg_type, flags, payload


# ── PSK authentication (mirrors greenboost_netc.c logic) ─────────────────────

_GB_KEY_PATH = "/etc/greenboost/cluster.key"

def _load_psk() -> bytes | None:
    """Return 32-byte PSK from cluster.key, or None if file is absent/malformed."""
    try:
        with open(_GB_KEY_PATH) as f:
            hex_str = f.readline().strip()
        if len(hex_str) < 64:
            return None
        return bytes.fromhex(hex_str[:64])
    except (OSError, ValueError):
        return None


def _do_psk_auth(sock) -> bool:
    """Perform PSK challenge-response if cluster.key is present.

    The server sends a 32-byte nonce first; we respond with
    HMAC-SHA256(psk, nonce).  If the key file is absent we skip auth
    (feeder must also have no key configured, or it will reject us).

    Returns True on success (or key absent), raises on auth failure.
    """
    psk = _load_psk()
    if psk is None:
        return True
    nonce = _recvall(sock, 32)
    mac = hmac.new(psk, nonce, hashlib.sha256).digest()
    sock.sendall(mac)
    return True


# ── Handshake ──────────────────────────────────────────────────────────────────

GB_NET_PROTO_VER      = 3
GB_NET_MAX_HOSTNAME   = 64
GB_NET_MAX_GPU_NAME   = 64
GB_NET_MAX_GPUS       = 8

# gb_net_gpu_info size: gpu_id(4) + vram_bytes(8) + cc_major(4) + cc_minor(4)
#                       + ram_available_bytes(8) + t3_bytes(8) + name(64) = 100 bytes
GB_GPU_INFO_SIZE      = 100

# GB_NET_FEAT_ZSTD (net_fabric.h): host advertises transparent zstd payload
# compression; feeder echoes it when built with libzstd.
GB_NET_FEAT_ZSTD = 1 << 0


def _build_handshake_req(feature_flags: int = 0):
    # gb_net_handshake_req: proto_version(4) gpu_count(4) hostname(64) gpus[8](800)
    #   = 872 bytes, plus a trailing feature_flags(4) (proto v3, optional).
    # netd accepts the 872-byte short form (feature_flags read as 0); append the
    # field only to negotiate a feature such as zstd compression.
    proto_version = GB_NET_PROTO_VER
    gpu_count = 0
    hostname = b"gb_diag\x00".ljust(GB_NET_MAX_HOSTNAME, b"\x00")
    gpus_blob = b"\x00" * (GB_NET_MAX_GPUS * GB_GPU_INFO_SIZE)
    req = struct.pack("<II", proto_version, gpu_count) + hostname + gpus_blob
    if feature_flags:
        req += struct.pack("<I", feature_flags)
    return req

def do_handshake(sock):
    payload = _build_handshake_req()
    _send_msg(sock, GB_MSG_HANDSHAKE_REQ, payload)
    msg_type, flags, resp = _recv_msg(sock)
    if msg_type not in (GB_MSG_HANDSHAKE_RESP, GB_MSG_RESPONSE):
        raise ValueError(f"Unexpected msg_type 0x{msg_type:02X} in handshake response")
    if len(resp) < 4:
        raise ValueError("Handshake response too short")
    status = struct.unpack_from("<I", resp, 0)[0]
    if status != GB_STATUS_OK:
        raise ValueError(f"Handshake rejected: status={status}")
    # Parse hostname from response (offset 12 after status+feeder_id+proto+gpu_count)
    feeder_hostname = ""
    if len(resp) >= 16 + GB_NET_MAX_HOSTNAME:
        raw_hn = resp[16:16 + GB_NET_MAX_HOSTNAME]
        raw_str = raw_hn.split(b"\x00")[0].decode("utf-8", errors="replace")
        if "�" in raw_str:
            print(f"[warn] feeder hostname contains non-UTF-8 bytes - replacement chars used: {raw_str!r}", file=sys.stderr)
        feeder_hostname = raw_str
    return feeder_hostname


# ── MEM_INFO query ────────────────────────────────────────────────────────────

def query_mem_info(sock, device_id=0):
    # gb_net_mem_info: device_id(4) t2_speed_mts(4)
    payload = struct.pack("<II", device_id, 0)
    _send_msg(sock, GB_MSG_MEM_INFO, payload)
    msg_type, flags, resp = _recv_msg(sock)
    if len(resp) < 24:
        return None
    # gb_net_mem_info_resp layout (little-endian):
    # status(4) t2_speed_mts(4) free_bytes(8) total_bytes(8)
    # t1_free(8) t1_total(8) t2_free(8) t2_total(8) t3_free(8) t3_total(8)
    # t3_speed_mbs(4) _pad3(4)
    if len(resp) >= 80:
        (status, t2_spd, free_b, total_b,
         t1_free, t1_total, t2_free, t2_total, t3_free, t3_total,
         t3_spd, _pad) = struct.unpack_from("<IIQQQQQQQQIi", resp, 0)
    elif len(resp) >= 56:
        (status, t2_spd, free_b, total_b,
         t1_free, t1_total, t2_free, t2_total) = struct.unpack_from("<IIQQQQQQ", resp, 0)
        t3_free = t3_total = t3_spd = 0
    else:
        (status, t2_spd, free_b, total_b) = struct.unpack_from("<IIQQ", resp, 0)
        t1_free = t1_total = t2_free = t2_total = t3_free = t3_total = t3_spd = 0

    return {
        "status": status, "t2_speed_mts": t2_spd,
        "t1_free": t1_free, "t1_total": t1_total,
        "t2_free": t2_free, "t2_total": t2_total,
        "t3_free": t3_free, "t3_total": t3_total,
        "t3_speed_mbs": t3_spd,
    }


# ── FEEDER_STATUS query (T1/T2/T3 + v3.1 GPU telemetry) ──────────────────────

def query_feeder_status(sock):
    """Send GB_MSG_FEEDER_STATUS (no payload) and parse gb_feeder_status_resp.

    Returns a dict with T1/T2/T3 free/total bytes, mps_sm_pct,
    kernel_dispatch_count, and (when the feeder's netd is v3.1+) live GPU
    telemetry: gpu_util_pct, gpu_temp_c, gpu_power_w, ecc_dbe_count,
    throttle_reasons. Telemetry fields are 0 on an older/short reply , this
    is the live-utilization counterpart to query_mem_info's static free/total.
    Returns None on a too-short or malformed reply.
    """
    _send_msg(sock, GB_MSG_FEEDER_STATUS)
    msg_type, flags, resp = _recv_msg(sock)
    if len(resp) < GB_FEEDER_STATUS_V30_SIZE:
        return None
    (status, mps_sm_pct,
     t1_free, t1_total, t2_free, t2_total, t3_free, t3_total,
     kernel_dispatch_count, _pad) = struct.unpack_from("<IIQQQQQQII", resp, 0)
    out = {
        "status": status, "mps_sm_pct": mps_sm_pct,
        "t1_free": t1_free, "t1_total": t1_total,
        "t2_free": t2_free, "t2_total": t2_total,
        "t3_free": t3_free, "t3_total": t3_total,
        "kernel_dispatch_count": kernel_dispatch_count,
        "gpu_temp_c": 0, "gpu_power_w": 0, "gpu_util_pct": 0,
        "ecc_dbe_count": 0, "throttle_reasons": 0,
    }
    if len(resp) >= GB_FEEDER_STATUS_V31_SIZE:
        (gpu_temp_c, gpu_power_w, gpu_util_pct, ecc_dbe_count,
         throttle_reasons, _pad2) = struct.unpack_from(
            "<HHIIII", resp, GB_FEEDER_STATUS_V30_SIZE)
        out.update(gpu_temp_c=gpu_temp_c, gpu_power_w=gpu_power_w,
                   gpu_util_pct=gpu_util_pct, ecc_dbe_count=ecc_dbe_count,
                   throttle_reasons=throttle_reasons)
    return out


def test_feeder_status(sock):
    _head("Feeder live status (GB_MSG_FEEDER_STATUS)")
    fs = query_feeder_status(sock)
    if not fs:
        _fail("FEEDER_STATUS query failed or response too short")
        return False
    _info(f"kernel_dispatch_count={fs['kernel_dispatch_count']}  "
          f"mps_sm_pct={fs['mps_sm_pct']}")
    if fs["gpu_util_pct"] or fs["gpu_temp_c"] or fs["gpu_power_w"]:
        _info(f"GPU util={fs['gpu_util_pct']}%  temp={fs['gpu_temp_c']}C  "
              f"power={fs['gpu_power_w']}W  ecc_dbe={fs['ecc_dbe_count']}  "
              f"throttle=0x{fs['throttle_reasons']:x}")
    else:
        _warn("v3.1 GPU telemetry fields are all zero , feeder netd may "
              "predate v3.1, or NVML util/temp/power queries failed")
    return True


# ── CUDA_MALLOC / CUDA_FREE ───────────────────────────────────────────────────

def alloc_on_feeder(sock, size_bytes, tier_flag, device_id=0):
    # gb_net_cuda_malloc: size(8) flags(4) device_id(4)
    payload = struct.pack("<QII", size_bytes, tier_flag, device_id)
    _send_msg(sock, GB_MSG_CUDA_MALLOC, payload)
    msg_type, flags, resp = _recv_msg(sock)
    if len(resp) < 4:
        return None, None, None
    # gb_net_cuda_malloc_resp: status(4) tier_used(4) remote_handle(8)
    if len(resp) >= 16:
        status, tier_used, handle = struct.unpack_from("<IIQ", resp, 0)
    else:
        status = struct.unpack_from("<I", resp, 0)[0]
        tier_used = 0xff
        handle = 0
    return status, tier_used, handle

def free_on_feeder(sock, remote_handle):
    # gb_net_cuda_free: remote_handle(8)
    payload = struct.pack("<Q", remote_handle)
    _send_msg(sock, GB_MSG_CUDA_FREE, payload)
    # No response expected / drain any response
    # Audit F-L5-01: catch only the exceptions actually possible here so
    # programming errors (KeyError, AttributeError) still surface.
    try:
        sock.settimeout(0.5)
        _recv_msg(sock)
    except (socket.timeout, OSError, EOFError, ValueError, struct.error):
        pass
    finally:
        sock.settimeout(10.0)


# ── Tests ─────────────────────────────────────────────────────────────────────

ALLOC_SIZE = 64 * 1024 * 1024  # 64 MB

def test_tier(sock, tier_flag, tier_label, feeder_ip=None):
    _head(f"Testing {tier_label} allocation")
    status, tier_used, handle = alloc_on_feeder(sock, ALLOC_SIZE, tier_flag)
    if status is None:
        _fail("No response from feeder")
        return False
    tier_str = TIER_NAMES.get(tier_used, f"tier={tier_used}")
    if status == GB_STATUS_OK:
        _ok(f"Allocated 64 MB  →  handle=0x{handle:016X}  tier_used={tier_str}")
        free_on_feeder(sock, handle)
        _info("Freed allocation")
        return True
    else:
        status_str = STATUS_NAMES.get(status, str(status))
        _fail(f"Allocation failed: status={status_str}  tier_used={tier_str}")
        if tier_flag == GB_ALLOC_TIER_T1 and status == GB_STATUS_ERR_OOM:
            _warn("T1 OOM - feeder CUDA context may not be initialized.")
            _warn("Restart feeder:  sudo greenboost feed stop && sudo greenboost feed start")
            ssh_prefix = f"ssh {feeder_ip} " if feeder_ip else "ssh <feeder> "
            _warn(f"Then check:      {ssh_prefix}'tail -20 /var/log/greenboost/netd.log'")
        return False

def test_mem_info(sock):
    _head("Feeder memory information")
    info = query_mem_info(sock)
    if not info:
        _fail("MEM_INFO query failed or response too short")
        return False
    t1f = info["t1_free"] >> 20; t1t = info["t1_total"] >> 20
    t2f = info["t2_free"] >> 20; t2t = info["t2_total"] >> 20
    t3f = info["t3_free"] >> 20; t3t = info["t3_total"] >> 20
    _info(f"T1 GPU VRAM : {t1f:6d} / {t1t:6d} MB free/total")
    _info(f"T2 DDR RAM  : {t2f:6d} / {t2t:6d} MB free/total  [{info['t2_speed_mts']} MT/s]")
    _info(f"T3 NVMe     : {t3f:6d} / {t3t:6d} MB free/total  [{info['t3_speed_mbs']} MB/s]")
    if t1t == 0:
        _warn("T1 total=0 - feeder may not have NVML available or CUDA not initialized")
    return True

def test_compute(sock):
    _head("Testing GPU compute dispatch (kernel exec)")
    # For compute test, allocate a tiny T1 block to get a handle, then try
    # sending a GB_MSG_CUDA_EXEC with that handle as an argument.
    status, tier_used, handle = alloc_on_feeder(sock, 4096, GB_ALLOC_TIER_T1)
    if status != GB_STATUS_OK:
        _fail("Cannot test compute - T1 allocation failed (fix T1 first)")
        return False

    # Build a minimal GB_MSG_CUDA_EXEC payload.
    # Wire format: gb_net_cuda_exec struct (48 bytes) first, then kernel_name bytes.
    # gb_net_cuda_exec fields (all u32, packed):
    #   grid_x/y/z (3×4), block_x/y/z (3×4), shared_mem_bytes (4),
    #   kernel_name_len (4), n_arg_vals (4), n_relocs (4), n_uploads (4), n_downloads (4)
    # Send a non-existent kernel and verify the feeder replies ERR_INVALID - that
    # confirms the compute dispatch channel is reachable.
    import struct as _s
    kernel_bytes = b"__gb_diag_test_kernel"
    exec_payload = (
        _s.pack("<IIIIIIIIIIII",
                1, 1, 1,              # grid_x/y/z
                1, 1, 1,              # block_x/y/z
                0,                    # shared_mem_bytes
                len(kernel_bytes),    # kernel_name_len
                0,                    # n_arg_vals
                0,                    # n_relocs
                0,                    # n_uploads
                0,                    # n_downloads
                ) +
        kernel_bytes
    )
    _send_msg(sock, GB_MSG_CUDA_EXEC, exec_payload)
    try:
        sock.settimeout(3.0)
        msg_type, flags, resp = _recv_msg(sock)
        sock.settimeout(10.0)
        # gb_net_response layout: u16 orig_msg_type + u16 _pad + u32 status
        # Status is at byte offset 4, NOT offset 0.
        if len(resp) >= 8:
            resp_status = struct.unpack_from("<I", resp, 4)[0]
        elif len(resp) >= 6:
            resp_status = struct.unpack_from("<H", resp, 4)[0]
        else:
            resp_status = 0xff
        # ERR_INVALID (kernel not found) means compute path reached feeder - good!
        if resp_status in (GB_STATUS_ERR_INVALID, GB_STATUS_OK):
            _ok(f"Feeder received exec request and replied (status={STATUS_NAMES.get(resp_status, resp_status)})")
            _info("Compute dispatch channel is working")
            result = True
        elif resp_status == GB_STATUS_ERR_CUDA:
            _warn("Feeder replied ERR_CUDA - CUDA context not initialized on feeder")
            _warn("Restart feeder after checking CUDA:  sudo systemctl restart greenboost-netd")
            result = False
        else:
            _warn(f"Feeder replied with unexpected status={resp_status}")
            result = False
    except (socket.timeout, OSError, EOFError, ValueError, struct.error) as e:
        # Audit F-L5-02: keep specific exceptions; let programming bugs raise.
        _fail(f"No response to exec request: {e}")
        result = False
    finally:
        sock.settimeout(10.0)

    free_on_feeder(sock, handle)
    return result


# ── N10: heartbeat latency test ───────────────────────────────────────────────

def test_heartbeat_latency(sock, n=100):
    _head(f"Heartbeat round-trip latency  ({n} pings)")
    latencies_ms = []
    ts_ms = int(time.time() * 1000)
    # gb_net_heartbeat wire payload: just timestamp_ms (8 bytes) as first field
    hb_payload = struct.pack("<Q", ts_ms) + b"\x00" * 8  # timestamp + pad
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            _send_msg(sock, GB_MSG_HEARTBEAT, hb_payload)
            sock.settimeout(2.0)
            _recv_msg(sock)
            sock.settimeout(10.0)
        except (socket.timeout, OSError, EOFError, ValueError, struct.error) as e:
            # Audit F-L5-21: keep specific exceptions in the heartbeat loop.
            _fail(f"Heartbeat failed mid-test: {e}")
            break
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    if not latencies_ms:
        _fail("No heartbeat responses received")
        return False
    s = sorted(latencies_ms)
    p50 = statistics.median(s)
    p95 = s[int(len(s) * 0.95)]
    p99 = s[int(len(s) * 0.99)] if len(s) >= 100 else s[-1]
    _ok(f"Received {len(latencies_ms)}/{n} heartbeat responses")
    _info(f"RTT p50={p50:.2f} ms   p95={p95:.2f} ms   p99={p99:.2f} ms")
    _info(f"RTT min={min(s):.2f} ms  max={max(s):.2f} ms")
    if p99 > 100:
        _warn(f"p99 latency {p99:.1f} ms is high - check feeder load or network quality")
    return True


# ── N10: rate-limit validation test ──────────────────────────────────────────

def test_rate_limit(sock):
    _head("Rate-limit validation  (flood CUDA_MALLOC → expect ERR_THROTTLE)")
    _info("Sending rapid malloc requests to exhaust 200-token bucket…")
    throttle_seen = False
    ok_count = 0
    handles = []
    # Send 300 rapid requests - bucket is 200, so throttle should appear by ~201
    for i in range(300):
        status, tier_used, handle = alloc_on_feeder(sock, 4096, GB_ALLOC_TIER_T2)
        if status is None:
            _fail(f"Connection lost at request {i+1}")
            break
        if status == GB_STATUS_ERR_THROTTLE:
            throttle_seen = True
            _ok(f"ERR_THROTTLE received at request {i+1}  (after {ok_count} successful allocs)")
            break
        if status == GB_STATUS_OK:
            ok_count += 1
            handles.append(handle)
        # After throttle seen, stop immediately
    # Clean up any successful allocs
    for h in handles:
        free_on_feeder(sock, h)
    if throttle_seen:
        _info(f"Rate limiter is working  ({ok_count} allocs before throttle)")
        return True
    else:
        _warn(f"ERR_THROTTLE was NOT seen after {ok_count} requests - "
              "rate limiter may be misconfigured or feeder does not implement N9")
        return False


# ── N10: multi-feeder concurrent stress test ──────────────────────────────────

def _run_one_feeder(ip, port):
    results = {}
    try:
        sock = socket.create_connection((ip, port), timeout=5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(10.0)
    except (OSError, socket.timeout) as e:
        # Audit F-L5-06: keep narrow exception types around connect.
        return ip, port, {"connect": False, "error": str(e)}
    try:
        _do_psk_auth(sock)
        _reset_seq()
        do_handshake(sock)
        results["T1"] = test_tier(sock, GB_ALLOC_TIER_T1, "T1 GPU VRAM", ip)
        results["T2"] = test_tier(sock, GB_ALLOC_TIER_T2, "T2 DDR", ip)
        results["T3"] = test_tier(sock, GB_ALLOC_TIER_T3, "T3 NVMe", ip)
        results["compute"] = test_compute(sock)
        results["telemetry"] = test_feeder_status(sock)
    except (socket.timeout, OSError, EOFError, ValueError, struct.error) as e:
        results["error"] = str(e)
    finally:
        # Audit F-L5-07: socket close in finally is correct.  Wrap in try
        # so a double-close doesn't propagate during pool teardown.
        try:
            sock.close()
        except OSError:
            pass
    return ip, port, results


def test_multi_feeder():
    _head("Multi-feeder concurrent stress test")
    feeders = read_all_cluster_conf()
    if not feeders:
        _fail("No feeders in /etc/greenboost/cluster.conf")
        return False
    _info(f"Found {len(feeders)} feeder(s) - running all tests concurrently")
    feeder_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(feeders)) as pool:
        futures = {pool.submit(_run_one_feeder, ip, port): (ip, port)
                   for ip, port in feeders}
        for fut in concurrent.futures.as_completed(futures):
            ip, port, results = fut.result()
            feeder_results[f"{ip}:{port}"] = results

    _head("Multi-feeder summary")
    all_pass = True
    col = 22
    print(f"  {'Feeder':<20}  {'T1':>4}  {'T2':>4}  {'T3':>4}  {'Compute':>8}  {'Telem':>6}")
    print(f"  {'─' * 60}")
    for addr, results in sorted(feeder_results.items()):
        def _s(k):
            v = results.get(k)
            if v is True:  return f"{C_GREEN}PASS{C_RESET}"
            if v is False: return f"{C_RED}FAIL{C_RESET}"
            return f"{C_YELLOW} -- {C_RESET}"
        err = results.get("error", "")
        err_str = f"  {C_RED}({err}){C_RESET}" if err else ""
        print(f"  {addr:<20}  {_s('T1'):>4}  {_s('T2'):>4}  {_s('T3'):>4}  "
              f"{_s('compute'):>8}  {_s('telemetry'):>6}{err_str}")
        if any(results.get(k) is False for k in ("T1", "T2", "T3", "compute", "telemetry")):
            all_pass = False
    print()
    return all_pass


# ── Read cluster.conf ─────────────────────────────────────────────────────────

def read_cluster_conf():
    entries = read_all_cluster_conf()
    if entries:
        return entries[0]
    return None, None


def read_all_cluster_conf():
    conf = "/etc/greenboost/cluster.conf"
    if not os.path.exists(conf):
        return []
    entries = []
    with open(conf) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                addr = parts[0]
                ip, _, port_str = addr.partition(":")
                # Audit F-L5-31: validate the port is numeric AND in range.
                # Falling back silently to 9740 on garbage hid malformed conf.
                if not port_str:
                    port = 9740
                elif port_str.isdigit():
                    port = int(port_str)
                    if not 1 <= port <= 65535:
                        sys.stderr.write(
                            f"[gb_feeder_diag] WARN: port {port} out of range "
                            f"for {ip}; using 9740\n"
                        )
                        port = 9740
                else:
                    sys.stderr.write(
                        f"[gb_feeder_diag] WARN: non-numeric port {port_str!r} "
                        f"for {ip}; using 9740\n"
                    )
                    port = 9740
                entries.append((ip, port))
    return entries


# ── Local GreenBoost diagnostic (no feeder needed) ───────────────────────────

def test_local() -> bool:
    """Check local GreenBoost kernel module, ioctl, and shim stats.

    Invoked by:  python3 gb_feeder_diag.py --local
    """
    _head("Local GreenBoost diagnostic")

    # Kernel module presence
    sysfs_mod = os.path.exists("/sys/module/greenboost")
    sysfs_cls = "/sys/class/greenboost/greenboost/status"
    dev       = "/dev/greenboost"

    if sysfs_mod:
        _ok("/sys/module/greenboost  (kernel module loaded)")
    else:
        _fail("/sys/module/greenboost missing - is the module loaded?")
        _info("Load with:  sudo modprobe greenboost")

    if os.path.exists(dev):
        _ok(f"{dev}  (character device accessible)")
    else:
        _warn(f"{dev} not found - ioctl interface unavailable")

    # Read sysfs class status
    _head("Sysfs class status  (/sys/class/greenboost/greenboost/status)")
    if os.path.exists(sysfs_cls):
        try:
            with open(sysfs_cls) as f:
                content = f.read()
            for line in content.strip().splitlines():
                _info(line)
        except OSError as e:
            _fail(f"Cannot read {sysfs_cls}: {e}")
    else:
        _warn(f"{sysfs_cls} not found")

    # Read shim stats
    _head("Shim stats  (/run/greenboost/shim_stats)")
    shim_found = False
    for shim_path in ["/run/greenboost/shim_stats", "/tmp/greenboost_shim_stats"]:
        if os.path.exists(shim_path):
            shim_found = True
            try:
                with open(shim_path) as f:
                    for line in f.read().strip().splitlines():
                        _info(line)
            except OSError as e:
                _fail(f"Cannot read {shim_path}: {e}")
            break
    if not shim_found:
        _warn("Shim stats not found - no active inference process using LD_PRELOAD shim")
        _info("Start with:  GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so <app>")

    # Check NVTX log
    _head("Recent NVTX events  (/run/greenboost/nvtx_events.log)")
    nvtx = "/run/greenboost/nvtx_events.log"
    if os.path.exists(nvtx):
        try:
            with open(nvtx) as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            for line in lines[-10:]:
                _info(line)
            if not lines:
                _warn("NVTX log empty")
        except OSError as e:
            _fail(f"Cannot read {nvtx}: {e}")
    else:
        _warn(f"{nvtx} not found - shim may not be active")

    return sysfs_mod


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GreenBoost feeder diagnostic tool")
    parser.add_argument("command", nargs="?", default="all",
                        choices=["t1", "t2", "t3", "compute", "telemetry", "all", "info"],
                        help="Which test to run (default: all)")
    parser.add_argument("--ip",   default=None, help="Feeder IP (overrides cluster.conf)")
    parser.add_argument("--port", type=int, default=None, help="Feeder port (default: 9740)")
    # N10: new diagnostic modes
    parser.add_argument("--heartbeat-latency", action="store_true",
                        help="N10: Send 100 heartbeat pings and report p50/p95/p99 RTT")
    parser.add_argument("--rate-limit", action="store_true",
                        help="N10: Flood CUDA_MALLOC requests and verify ERR_THROTTLE is returned")
    parser.add_argument("--multi-feeder", action="store_true",
                        help="N10: Run all tests against all feeders in cluster.conf concurrently")
    parser.add_argument("--local", action="store_true",
                        help="Test local GreenBoost kernel module and shim (no feeder required)")
    args = parser.parse_args()

    # --local: test the local kernel module + shim without connecting to a feeder
    if args.local:
        print(f"\n{C_BOLD}GreenBoost Local Diagnostic{C_RESET}")
        print(f"{C_DIM}{'─' * 50}{C_RESET}")
        ok = test_local()
        sys.exit(0 if ok else 1)

    # N10 --multi-feeder is cluster-wide; handle separately before single-feeder path
    if args.multi_feeder:
        print(f"\n{C_BOLD}GreenBoost Multi-Feeder Diagnostic{C_RESET}")
        print(f"{C_DIM}{'─' * 50}{C_RESET}")
        ok = test_multi_feeder()
        sys.exit(0 if ok else 1)

    ip, port = args.ip, args.port
    if not ip:
        ip, port = read_cluster_conf()
    if not ip:
        print(f"{C_RED}No feeder configured.{C_RESET}  Run: sudo greenboost connect <IP>")
        sys.exit(1)
    if not port:
        port = 9740

    print(f"\n{C_BOLD}GreenBoost Feeder Diagnostic{C_RESET}  →  {ip}:{port}")
    print(f"{C_DIM}{'─' * 50}{C_RESET}")

    try:
        sock = socket.create_connection((ip, port), timeout=5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(10.0)
    except Exception as e:
        _fail(f"Cannot connect to feeder at {ip}:{port}  ({e})")
        _info("Is the feeder running?  sudo systemctl status greenboost-netd")
        sys.exit(1)

    _ok(f"Connected to {ip}:{port}")

    try:
        _do_psk_auth(sock)
        _ok(f"PSK auth OK  (key={'present' if _load_psk() else 'absent - unauthenticated'})")
    except Exception as e:
        _fail(f"PSK auth failed: {e}")
        sock.close()
        sys.exit(1)

    _reset_seq()

    try:
        feeder_host = do_handshake(sock)
        _ok(f"Handshake OK  (feeder hostname: {feeder_host or '?'})")
    except Exception as e:
        _fail(f"Handshake failed: {e}")
        sock.close()
        sys.exit(1)

    results = {}

    def _run_check(name, fn, *fn_args):
        # One hung/erroring check (e.g. a GB_MSG_FEEDER_STATUS timeout) must
        # not crash the whole `all` run with a raw traceback , that hides
        # whether every OTHER check (notably GPU Compute) actually passed.
        # Each check gets its own failure reported in the summary instead.
        try:
            results[name] = fn(*fn_args)
        except Exception as e:
            _fail(f"{name} check raised {e.__class__.__name__}: {e}")
            results[name] = False

    try:
        test_mem_info(sock)

        # N10: extended diagnostic modes - exclusive; skip standard suite when used
        if args.heartbeat_latency:
            _run_check("Heartbeat RTT", test_heartbeat_latency, sock)
        elif args.rate_limit:
            _run_check("Rate Limit", test_rate_limit, sock)
        else:
            if args.command in ("t1", "all"):
                _run_check("T1 VRAM", test_tier, sock, GB_ALLOC_TIER_T1, "T1 GPU VRAM", ip)
            if args.command in ("t2", "all"):
                _run_check("T2 DDR", test_tier, sock, GB_ALLOC_TIER_T2, "T2 DDR (pinned)", ip)
            if args.command in ("t3", "all"):
                _run_check("T3 NVMe", test_tier, sock, GB_ALLOC_TIER_T3, "T3 NVMe/pageable", ip)
            if args.command in ("compute", "all"):
                _run_check("GPU Compute", test_compute, sock)
            if args.command in ("telemetry", "all"):
                _run_check("Telemetry", test_feeder_status, sock)
    finally:
        sock.close()

    if results:
        _head("Summary")
        all_pass = True
        for name, ok in results.items():
            if ok:
                _ok(f"{name:20s}  PASS")
            else:
                _fail(f"{name:20s}  FAIL")
                all_pass = False
        print()
        if all_pass:
            print(f"  {C_GREEN}{C_BOLD}All tests passed.{C_RESET}  Feeder T1/T2/T3 and compute are working.\n")
        else:
            print(f"  {C_YELLOW}{C_BOLD}Some tests failed.{C_RESET}  "
                  f"Check feeder logs:  ssh {ip} 'tail -40 /var/log/greenboost/netd.log'\n")
        sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
