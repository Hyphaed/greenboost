#!/usr/bin/python3 -I
"""GreenBoost root helper: a closed list of verbs behind a unix socket.

Design: docs/greenboost-audit/privsplit-design.md (SP41-12, QC1).

The unprivileged brain asks, this process acts. It never decides policy and
never runs anything the verb table does not name. Rules enforced here:

* one request and one reply per connection; 4-byte big-endian length plus one
  JSON object, at most MAX_FRAME bytes; any other framing is refused;
* the peer uid comes from SO_PEERCRED (kernel-recorded at connect()) and must
  be in the allow-list; a pid from the request body is never used;
* every argument is validated against a fixed schema (enums, ints in range, a
  strict bdf regex); no argument is a path, a shell string or a binary name;
* privileged work goes through a Backend. SystemBackend builds argv lists with
  absolute binary paths and runs them with shell=False and a minimal
  environment; tests substitute a recording backend;
* every request and reply is one JSON line in the audit sink.

Standard library only. Keep this file under 400 lines of logic.
"""
from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import socket
import struct
import subprocess
import sys
import time

MAX_FRAME = 4096
CONN_TIMEOUT_S = 5.0
CODES = ("ok", "bad-verb", "bad-arg", "peer-denied", "precondition", "rate-limited", "failed")

NVIDIA_UNLOAD_ORDER = ("nvidia_drm", "nvidia_modeset", "nvidia_uvm", "nvidia")  # dependents first
LOADABLE = ("greenboost",) + tuple(reversed(NVIDIA_UNLOAD_ORDER))
MRRS_CODE = {128: 0, 256: 1, 512: 2, 1024: 3, 2048: 4, 4096: 5}
GOVERNORS = ("powersave", "performance", "schedutil")
VM_KEYS = {  # key -> (lo, hi)
    "vm.swappiness": (0, 200), "vm.dirty_ratio": (1, 90), "vm.dirty_background_ratio": (1, 90),
    "vm.vfs_cache_pressure": (1, 500), "vm.min_free_kbytes": (1024, 4194304),
}
BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
EPP_RE = re.compile(r"^[a-z_]{1,32}$")
# verb -> (max calls, per seconds)
RATE = {"module-load": (3, 60), "module-unload": (2, 60), "pcie-maxreadreq": (3, 60),
        "gpu-clocks": (12, 60), "gpu-power": (12, 60), "cpu-governor": (12, 60),
        "vm-sysctl": (30, 60), "drop-caches": (6, 60), "apparmor-reload": (2, 60),
        "read-gpu-config": (12, 60)}


class ArgError(ValueError):
    pass


class Precondition(Exception):
    pass


class BackendError(Exception):
    pass


# ---- validators: each returns the normalized value or raises ArgError ----
def _enum(*vals):
    def v(x):
        if type(x) is not type(vals[0]) or x not in vals:
            raise ArgError("value not allowed")
        return x
    return v


def _int(lo, hi):
    def v(x):
        if type(x) is not int or not lo <= x <= hi:
            raise ArgError("integer out of range")
        return x
    return v


def _bdf(x):
    if not isinstance(x, str) or not BDF_RE.fullmatch(x):
        raise ArgError("bad bdf")
    return x


def _epp(x):
    if not isinstance(x, str) or not EPP_RE.fullmatch(x):
        raise ArgError("bad epp")
    return x


def _unload_names(x):
    if not isinstance(x, list) or not 1 <= len(x) <= len(NVIDIA_UNLOAD_ORDER):
        raise ArgError("names must be a short list")
    idx = []
    for n in x:
        if not isinstance(n, str) or n not in NVIDIA_UNLOAD_ORDER:
            raise ArgError("module not allowed")
        idx.append(NVIDIA_UNLOAD_ORDER.index(n))
    if idx != sorted(set(idx)):
        raise ArgError("unload order wrong or duplicate")
    return list(x)


def _vm_value(args):
    lo, hi = VM_KEYS[args["key"]]
    _int(lo, hi)(args["value"])


SCHEMAS = {
    "module-load": {"name": _enum(*LOADABLE)},
    "module-unload": {"names": _unload_names},
    "pcie-maxreadreq": {"bdf": _bdf, "value": _enum(*MRRS_CODE)},
    "gpu-clocks": {"min_mhz": _int(0, 10000), "max_mhz": _int(0, 10000)},
    "gpu-power": {"watts": _int(1, 2000)},
    "cpu-governor": {"governor": _enum(*GOVERNORS), "epp": _epp},
    "vm-sysctl": {"key": _enum(*VM_KEYS), "value": _int(-(2 ** 31), 2 ** 31)},
    "drop-caches": {"level": _enum(1, 2, 3)},
    "apparmor-reload": {},
    "read-gpu-config": {},
}


def validate(verb: str, args) -> dict:
    schema = SCHEMAS[verb]
    if not isinstance(args, dict) or set(args) != set(schema):
        raise ArgError("args keys do not match schema")
    out = {k: schema[k](args[k]) for k in schema}
    if verb == "vm-sysctl":
        _vm_value(out)
    if verb == "gpu-clocks" and out["min_mhz"] > out["max_mhz"]:
        raise ArgError("min above max")
    return out


def mrrs_new_value(cur: int, nbytes: int) -> int:
    """DevCtl bits 14:12 hold MaxReadReq."""
    return (cur & ~(0x7 << 12)) | (MRRS_CODE[nbytes] << 12)


def peer_allowed(creds, allowed_uids) -> bool:
    """creds is (pid, uid, gid) as read from SO_PEERCRED."""
    return creds is not None and creds[1] in allowed_uids


def cgroup_matches(pid: int, unit: str, proc_root: str = "/proc") -> bool:
    """Advisory only (pid reuse): the uid check is the gate."""
    try:
        with open(f"{proc_root}/{int(pid)}/cgroup", encoding="ascii", errors="replace") as f:
            return unit in f.read()
    except (OSError, ValueError):
        return False


# ---- backends ----
class Backend:
    """Interface. Every privileged or system-reading action lives here."""
    def module_load(self, name): raise NotImplementedError
    def module_unload(self, names): raise NotImplementedError
    def nvidia_in_use(self) -> bool: raise NotImplementedError
    def gpu_bdfs(self) -> list: raise NotImplementedError
    def pcie_maxreadreq(self, bdf, value) -> str: raise NotImplementedError
    def gpu_limits(self) -> dict: raise NotImplementedError   # clock_min/max, power_min_w/max_w
    def gpu_clocks(self, lo, hi): raise NotImplementedError
    def gpu_power(self, watts): raise NotImplementedError
    def cpu_offered(self) -> dict: raise NotImplementedError  # governors, epp (lists)
    def cpu_set(self, governor, epp): raise NotImplementedError
    def vm_sysctl(self, key, value): raise NotImplementedError
    def drop_caches(self, level): raise NotImplementedError
    def apparmor_profile_ok(self) -> bool: raise NotImplementedError
    def apparmor_reload(self): raise NotImplementedError
    def read_gpu_config(self) -> str: raise NotImplementedError


def _find(name, dirs=("/usr/sbin", "/sbin", "/usr/bin", "/bin")):
    for d in dirs:
        p = f"{d}/{name}"
        if os.access(p, os.X_OK):
            return p
    return f"/usr/sbin/{name}"


def _default_runner(argv, timeout):
    """The one place a process is started: argv list, shell off, tiny env."""
    r = subprocess.run(argv, shell=False, capture_output=True, text=True, timeout=timeout,
                       env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"}, check=False)
    return r.returncode, r.stdout, r.stderr


class SystemBackend(Backend):
    PROFILE = "/etc/apparmor.d/usr.lib.snapd.snap-confine.real"
    CPU = "/sys/devices/system/cpu"

    def __init__(self, runner=_default_runner):
        self.run = runner

    def _do(self, argv, timeout=20):
        assert isinstance(argv, list) and argv[0].startswith("/") and all(isinstance(a, str) for a in argv)
        try:
            rc, out, err = self.run(argv, timeout)
        except subprocess.TimeoutExpired:
            raise BackendError("timeout")
        except OSError as e:
            raise BackendError(f"exec failed: {e.strerror}")
        if rc != 0:
            raise BackendError(f"{os.path.basename(argv[0])} exit {rc}: {err.strip()[:120]}")
        return out

    def module_load(self, name):
        self._do([_find("modprobe"), "--", name], 20)

    def module_unload(self, names):
        self._do([_find("rmmod"), "--", *names], 20)

    def nvidia_in_use(self):
        # /sys/module/<m>/refcnt is world-readable (no CAP_SYS_PTRACE needed); rmmod
        # without -f refuses anyway, so this is a cheap early refusal.
        for m in NVIDIA_UNLOAD_ORDER:
            try:
                with open(f"/sys/module/{m}/refcnt", encoding="ascii") as f:
                    if int(f.read().strip() or 0) > 0:
                        return True
            except (OSError, ValueError):
                continue
        return False

    def gpu_bdfs(self):
        out = []
        base = "/sys/bus/pci/devices"
        try:
            for d in sorted(os.listdir(base)):
                with open(f"{base}/{d}/vendor", encoding="ascii") as v, open(f"{base}/{d}/class", encoding="ascii") as c:
                    if v.read().strip() == "0x10de" and c.read().strip().startswith("0x03"):
                        out.append(d)
        except OSError:
            pass
        return out

    def pcie_maxreadreq(self, bdf, value):
        setpci = _find("setpci")
        cur = int(self._do([setpci, "-s", bdf, "CAP_EXP+8.w"], 10).strip(), 16)
        new = mrrs_new_value(cur, value)
        self._do([setpci, "-s", bdf, f"CAP_EXP+8.w={new:04x}"], 10)
        after = int(self._do([setpci, "-s", bdf, "CAP_EXP+8.w"], 10).strip(), 16)
        if after != new:
            raise BackendError("read back differs")
        return f"{cur:04x}->{after:04x}"

    def gpu_limits(self):
        q = self._do([_find("nvidia-smi"), "--query-gpu=clocks.min_gr,clocks.max_gr,power.min_limit,power.max_limit",
                      "--format=csv,noheader,nounits"], 10)
        a = [float(x) for x in q.splitlines()[0].split(",")]
        return {"clock_min": int(a[0]), "clock_max": int(a[1]), "power_min_w": int(a[2]), "power_max_w": int(a[3])}

    def gpu_clocks(self, lo, hi):
        self._do([_find("nvidia-smi"), "-lgc", f"{lo},{hi}"], 15)

    def gpu_power(self, watts):
        self._do([_find("nvidia-smi"), "-pl", str(int(watts))], 15)

    def _cpu_file(self, name):
        with open(f"{self.CPU}/cpu0/cpufreq/{name}", encoding="ascii") as f:
            return f.read().split()

    def cpu_offered(self):
        try:
            return {"governors": self._cpu_file("scaling_available_governors"),
                    "epp": self._cpu_file("energy_performance_available_preferences")}
        except OSError:
            return {"governors": [], "epp": []}

    def cpu_set(self, governor, epp):
        try:
            for d in sorted(os.listdir(self.CPU)):
                if re.fullmatch(r"cpu[0-9]+", d):
                    for name, val in (("scaling_governor", governor), ("energy_performance_preference", epp)):
                        p = f"{self.CPU}/{d}/cpufreq/{name}"
                        if os.path.exists(p):
                            with open(p, "w", encoding="ascii") as f:
                                f.write(val)
        except OSError as e:
            raise BackendError(f"cpufreq write: {e.strerror}")

    def vm_sysctl(self, key, value):
        try:
            with open("/proc/sys/vm/" + key.split(".", 1)[1], "w", encoding="ascii") as f:
                f.write(str(value))
        except OSError as e:
            raise BackendError(f"sysctl write: {e.strerror}")

    def drop_caches(self, level):
        self.vm_sysctl("vm.drop_caches", level)

    def apparmor_profile_ok(self):
        try:
            st = os.stat(self.PROFILE)
        except OSError:
            return False
        return st.st_uid == 0 and not st.st_mode & 0o022

    def apparmor_reload(self):
        self._do([_find("apparmor_parser"), "-r", self.PROFILE], 15)
        self._do([_find("systemctl"), "restart", "apparmor.service"], 30)

    def read_gpu_config(self):
        res = {}
        for bdf in self.gpu_bdfs():
            try:
                res[bdf] = self._do([_find("setpci"), "-s", bdf, "CAP_EXP+8.w"], 10).strip()
            except BackendError as e:
                res[bdf] = f"error: {e}"
        return json.dumps(res, sort_keys=True)


# ---- the helper proper ----
class Helper:
    def __init__(self, backend, allowed_uids, audit=None, clock=time.monotonic, wall=time.time,
                 expected_unit="greenboost-brain.service", rate=RATE):
        self.b, self.allowed, self.clock, self.wall = backend, frozenset(allowed_uids), clock, wall
        self.audit_sink = audit or (lambda line: sys.stderr.write(line + "\n"))
        self.unit, self.rate = expected_unit, rate
        self.calls: dict = {}
        self.last_drop = None
        self.stop = False

    def _audit(self, creds, verb, args, rid, code, detail, cg=None):
        pid, uid, gid = creds if creds else (None, None, None)
        v = verb if isinstance(verb, str) and len(verb) <= 40 else None
        line = json.dumps({"ts": round(self.wall(), 3), "peer_pid": pid, "peer_uid": uid, "peer_gid": gid,
                           "verb": v, "args": args, "id": rid, "code": code, "detail": detail[:200],
                           "cgroup_ok": cg}, separators=(",", ":"), sort_keys=True)
        try:
            self.audit_sink(line)
        except Exception:
            pass

    def _rate_ok(self, verb) -> bool:
        n, win = self.rate[verb]
        now = self.clock()
        q = [t for t in self.calls.get(verb, []) if now - t < win]
        if len(q) >= n:
            self.calls[verb] = q
            return False
        q.append(now)
        self.calls[verb] = q
        return True

    def handle(self, creds, body: bytes) -> dict:
        """Pure request handling: creds is (pid, uid, gid), body the JSON bytes."""
        rid = None
        def reply(ok, code, detail, verb=None, args=None):
            self._audit(creds, verb, args, rid, code, detail, cgroup_matches(creds[0], self.unit) if creds else None)
            return {"ok": ok, "code": code, "detail": detail, "id": rid}
        if not peer_allowed(creds, self.allowed):
            return reply(False, "peer-denied", "uid not allowed")
        if len(body) > MAX_FRAME:
            return reply(False, "bad-arg", "oversized")
        try:
            req = json.loads(body.decode("utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        except (ValueError, UnicodeDecodeError, RecursionError):
            return reply(False, "bad-arg", "malformed json")
        if not isinstance(req, dict) or set(req) != {"verb", "args", "id"}:
            return reply(False, "bad-arg", "request keys must be verb, args, id")
        if isinstance(req["id"], str) and ID_RE.fullmatch(req["id"]):
            rid = req["id"]
        else:
            return reply(False, "bad-arg", "bad id")
        verb = req["verb"]
        if not isinstance(verb, str) or verb not in SCHEMAS:
            return reply(False, "bad-verb", "unknown verb", verb)
        try:
            args = validate(verb, req["args"])
        except ArgError as e:
            return reply(False, "bad-arg", str(e), verb)
        if not self._rate_ok(verb):
            return reply(False, "rate-limited", "too many calls", verb, args)
        try:
            detail = self._exec(verb, args)
        except Precondition as e:
            return reply(False, "precondition", str(e), verb, args)
        except BackendError as e:
            return reply(False, "failed", str(e), verb, args)
        except Exception:
            return reply(False, "failed", "internal error", verb, args)
        return reply(True, "ok", detail, verb, args)

    def _exec(self, verb, a) -> str:
        b = self.b
        if verb == "module-load":
            b.module_load(a["name"])
        elif verb == "module-unload":
            if b.nvidia_in_use():
                raise Precondition("nvidia device in use")
            b.module_unload(a["names"])
        elif verb == "pcie-maxreadreq":
            if a["bdf"] not in b.gpu_bdfs():
                raise Precondition("bdf is not an NVIDIA GPU")
            return b.pcie_maxreadreq(a["bdf"], a["value"])
        elif verb == "gpu-clocks":
            lim = b.gpu_limits()
            if not (lim["clock_min"] <= a["min_mhz"] <= a["max_mhz"] <= lim["clock_max"]):
                raise Precondition("clocks outside card range")
            b.gpu_clocks(a["min_mhz"], a["max_mhz"])
        elif verb == "gpu-power":
            lim = b.gpu_limits()
            if not lim["power_min_w"] <= a["watts"] <= lim["power_max_w"]:
                raise Precondition("power outside card range")
            b.gpu_power(a["watts"])
        elif verb == "cpu-governor":
            off = b.cpu_offered()
            if a["governor"] not in off["governors"] or a["epp"] not in off["epp"]:
                raise Precondition("not offered by the kernel")
            b.cpu_set(a["governor"], a["epp"])
        elif verb == "vm-sysctl":
            b.vm_sysctl(a["key"], a["value"])
        elif verb == "drop-caches":
            now = self.clock()
            if self.last_drop is not None and now - self.last_drop < 60:
                raise Precondition("drop-caches at most once per 60 s")
            self.last_drop = now
            b.drop_caches(a["level"])
        elif verb == "apparmor-reload":
            if not b.apparmor_profile_ok():
                raise Precondition("profile not root-owned")
            b.apparmor_reload()
        elif verb == "read-gpu-config":
            return b.read_gpu_config()
        return "done"

    # ---- socket layer ----
    @staticmethod
    def peercred(conn):
        raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return struct.unpack("3i", raw)

    def serve_conn(self, conn):
        conn.settimeout(CONN_TIMEOUT_S)
        creds = None
        try:
            creds = self.peercred(conn)
            if not peer_allowed(creds, self.allowed):
                self._send(conn, self.handle(creds, b""))
                return
            hdr = self._recv(conn, 4)
            (n,) = struct.unpack(">I", hdr)
            if n > MAX_FRAME:
                # refuse without reading the body: close after the reply
                self._send(conn, self.handle(creds, b"x" * (MAX_FRAME + 1)))
                return
            self._send(conn, self.handle(creds, self._recv(conn, n)))
        except (OSError, ConnectionError, struct.error):
            self._audit(creds, None, None, None, "bad-arg", "framing error or timeout")
        finally:
            conn.close()

    @staticmethod
    def _recv(conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("short read")
            buf += chunk
        return buf

    @staticmethod
    def _send(conn, obj):
        data = json.dumps(obj, separators=(",", ":")).encode()
        conn.sendall(struct.pack(">I", len(data)) + data)

    def serve_forever(self, listener, tick=None):
        listener.settimeout(0.2)
        while not self.stop:
            if tick:
                tick()
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            self.serve_conn(conn)


def sd_notify(msg: str) -> None:
    path = os.environ.get("NOTIFY_SOCKET")
    if not path:
        return
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        s.connect("\0" + path[1:] if path.startswith("@") else path)
        s.send(msg.encode())
    except OSError:
        pass
    finally:
        s.close()


def listener_from_env_or_path(path):
    if os.environ.get("LISTEN_PID") == str(os.getpid()) and os.environ.get("LISTEN_FDS") == "1":
        return socket.socket(fileno=3)
    if not path:
        raise SystemExit("no socket: run under the .socket unit or pass --socket")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    try:
        s.bind(path)
    finally:
        os.umask(old)
    s.listen(4)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--socket", default=None)
    ap.add_argument("--allow-user", action="append", default=[])
    ap.add_argument("--allow-uid", action="append", type=int, default=[])
    a = ap.parse_args(argv)
    uids = set(a.allow_uid)
    for u in a.allow_user:
        uids.add(pwd.getpwnam(u).pw_uid)
    if not uids:
        raise SystemExit("refusing to start with an empty uid allow-list")
    h = Helper(SystemBackend(), uids)
    lst = listener_from_env_or_path(a.socket)
    sd_notify("READY=1\n")
    last = [0.0]
    def tick():
        if time.monotonic() - last[0] > 5:
            last[0] = time.monotonic()
            sd_notify("WATCHDOG=1\n")
    h.serve_forever(lst, tick)


if __name__ == "__main__":
    main()
