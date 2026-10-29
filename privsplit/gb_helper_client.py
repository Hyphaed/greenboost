"""Client for the GreenBoost root helper. Runs in the unprivileged brain.

One connection per request. On any transport problem the client returns a
`failed` reply with detail `helper-unavailable`; it never falls back to
running rmmod, modprobe or setpci itself (the brain degrades to observe-only).
Standard library only.
"""
from __future__ import annotations

import itertools
import json
import os
import socket
import struct

DEFAULT_SOCKET = "/run/greenboost-helper/root-helper.sock"
MAX_FRAME = 4096
_seq = itertools.count(1)


class HelperClient:
    def __init__(self, path: str | None = None, timeout: float = 40.0):
        self.path = path or os.environ.get("GB_ROOT_HELPER_SOCKET", DEFAULT_SOCKET)
        self.timeout = timeout

    def call(self, verb: str, **args) -> dict:
        rid = f"{os.getpid()}-{next(_seq)}"
        body = json.dumps({"verb": verb, "args": args, "id": rid}, separators=(",", ":")).encode()
        if len(body) > MAX_FRAME:
            return {"ok": False, "code": "bad-arg", "detail": "request too large", "id": rid}
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.path)
            s.sendall(struct.pack(">I", len(body)) + body)
            (n,) = struct.unpack(">I", self._recv(s, 4))
            if n > MAX_FRAME:
                raise ConnectionError("oversized reply")
            rep = json.loads(self._recv(s, n).decode("utf-8"))
            if not isinstance(rep, dict) or "ok" not in rep or "code" not in rep:
                raise ValueError("bad reply")
            return rep
        except (OSError, ConnectionError, ValueError, struct.error):
            return {"ok": False, "code": "failed", "detail": "helper-unavailable", "id": rid}
        finally:
            s.close()

    @staticmethod
    def _recv(s, n):
        buf = b""
        while len(buf) < n:
            c = s.recv(n - len(buf))
            if not c:
                raise ConnectionError("short read")
            buf += c
        return buf

    # one method per verb, same names as the verbs
    def module_load(self, name): return self.call("module-load", name=name)
    def module_unload(self, names): return self.call("module-unload", names=list(names))
    def pcie_maxreadreq(self, bdf, value): return self.call("pcie-maxreadreq", bdf=bdf, value=value)
    def gpu_clocks(self, min_mhz, max_mhz): return self.call("gpu-clocks", min_mhz=min_mhz, max_mhz=max_mhz)
    def gpu_power(self, watts): return self.call("gpu-power", watts=watts)
    def cpu_governor(self, governor, epp): return self.call("cpu-governor", governor=governor, epp=epp)
    def vm_sysctl(self, key, value): return self.call("vm-sysctl", key=key, value=value)
    def drop_caches(self, level): return self.call("drop-caches", level=level)
    def apparmor_reload(self): return self.call("apparmor-reload")
    def read_gpu_config(self): return self.call("read-gpu-config")
