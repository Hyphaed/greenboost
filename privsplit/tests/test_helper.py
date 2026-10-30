"""Unprivileged tests of the root helper. Every privileged action is stubbed."""
import ast
import json
import os
import random
import socket
import string
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import gb_helper_client as client  # noqa: E402
import gb_root_helper as rh  # noqa: E402

CACHE = Path.home() / ".cache" / "gb-privsplit-tests"
CACHE.mkdir(parents=True, exist_ok=True)
ME = os.getuid()


class Rec(rh.Backend):
    """Recording backend: nothing here touches the system."""
    def __init__(self):
        self.calls = []
        self.in_use = False
        self.bdfs = ["0000:01:00.0"]
        self.limits = {"clock_min": 210, "clock_max": 3000, "power_min_w": 100, "power_max_w": 450}
        self.offered = {"governors": ["powersave", "performance"], "epp": ["performance", "power"]}
        self.profile_ok = True
        self.fail = False

    def _r(self, name, *a):
        if self.fail:
            raise rh.BackendError("boom")
        self.calls.append((name,) + a)
        return "stub"

    def module_load(self, n): self._r("module_load", n)
    def module_unload(self, n): self._r("module_unload", tuple(n))
    def nvidia_in_use(self): return self.in_use
    def gpu_bdfs(self): return self.bdfs
    def pcie_maxreadreq(self, b, v): return self._r("pcie", b, v)
    def gpu_limits(self): return self.limits
    def gpu_clocks(self, lo, hi): self._r("clocks", lo, hi)
    def gpu_power(self, w): self._r("power", w)
    def cpu_offered(self): return self.offered
    def cpu_set(self, g, e): self._r("cpu", g, e)
    def vm_sysctl(self, k, v): self._r("vm", k, v)
    def drop_caches(self, l): self._r("drop", l)
    def apparmor_profile_ok(self): return self.profile_ok
    def apparmor_reload(self): self._r("apparmor")
    def read_gpu_config(self): return self._r("readcfg")


def mk(backend=None, uids=(ME,), rate=None, clock=None):
    lines = []
    b = backend or Rec()
    t = [1000.0]
    h = rh.Helper(b, uids, audit=lines.append, clock=clock or (lambda: t[0]),
                  rate=rate or rh.RATE)
    h._t = t
    return h, b, lines


def req(verb, args=None, rid="r1"):
    return json.dumps({"verb": verb, "args": {} if args is None else args, "id": rid}).encode()


CREDS = (os.getpid(), ME, os.getgid())


class Protocol(unittest.TestCase):
    def test_unknown_verb_refused(self):
        h, b, _ = mk()
        for v in ("rm", "exec", "module_load", "", "MODULE-LOAD", 5, None):
            r = h.handle(CREDS, json.dumps({"verb": v, "args": {}, "id": "a"}).encode())
            self.assertEqual(r["code"], "bad-verb", v)
        self.assertEqual(b.calls, [])

    def test_bad_schema_refused(self):
        h, b, _ = mk()
        bad = [
            ("module-load", {}), ("module-load", {"name": "evil"}), ("module-load", {"name": "../x"}),
            ("module-load", {"name": "nvidia", "extra": 1}), ("module-load", {"name": ["nvidia"]}),
            ("module-load", {"name": "nvidia\x00"}), ("module-load", {"name": "/lib/x.ko"}),
            ("module-unload", {"names": ["greenboost"]}), ("module-unload", {"names": ["nvidia", "nvidia_drm"]}),
            ("module-unload", {"names": []}), ("module-unload", {"names": ["nvidia", "nvidia"]}),
            ("module-unload", {"names": "nvidia"}),
            ("pcie-maxreadreq", {"bdf": "0000:01:00.0; reboot", "value": 4096}),
            ("pcie-maxreadreq", {"bdf": "0000:01:00.0", "value": 300}),
            ("pcie-maxreadreq", {"bdf": "0000:01:00.0", "value": "4096"}),
            ("pcie-maxreadreq", {"bdf": "0000:01:00.0", "value": True}),
            ("gpu-clocks", {"min_mhz": 500, "max_mhz": 400}), ("gpu-clocks", {"min_mhz": 1.5, "max_mhz": 2}),
            ("gpu-power", {"watts": -1}), ("gpu-power", {"watts": True}),
            ("cpu-governor", {"governor": "ondemand", "epp": "power"}),
            ("cpu-governor", {"governor": "powersave", "epp": "power; id"}),
            ("vm-sysctl", {"key": "kernel.core_pattern", "value": 1}),
            ("vm-sysctl", {"key": "vm.swappiness", "value": 1000}),
            ("vm-sysctl", {"key": "vm.swappiness", "value": "10"}),
            ("drop-caches", {"level": 4}), ("drop-caches", {"level": "1"}),
            ("apparmor-reload", {"profile": "/etc/passwd"}),
            ("read-gpu-config", {"x": 1}),
        ]
        for verb, args in bad:
            r = h.handle(CREDS, req(verb, args))
            self.assertEqual(r["code"], "bad-arg", (verb, args))
        for body in (b"not json", b"[]", b'{"verb":"drop-caches"}', b'{"verb":"drop-caches","args":{"level":1},"id":"a","x":1}',
                     b'{"verb":"drop-caches","args":[],"id":"a"}', b'{"verb":"drop-caches","args":{"level":NaN},"id":"a"}',
                     req("drop-caches", {"level": 1}, rid="bad id!"), b"\xff\xfe"):
            self.assertEqual(h.handle(CREDS, body)["code"], "bad-arg", body)
        self.assertEqual(b.calls, [])

    def test_wrong_uid_refused(self):
        h, b, lines = mk(uids=(ME + 1,))
        r = h.handle(CREDS, req("drop-caches", {"level": 1}))
        self.assertEqual(r["code"], "peer-denied")
        self.assertIsNone(h.handle(None, req("drop-caches", {"level": 1}))["ok"] or None)
        self.assertEqual(b.calls, [])
        self.assertEqual(json.loads(lines[0])["peer_uid"], ME)

    def test_oversized_refused(self):
        h, b, _ = mk()
        big = req("vm-sysctl", {"key": "vm.swappiness", "value": 1}, rid="a" * 64) + b" " * rh.MAX_FRAME
        self.assertEqual(h.handle(CREDS, big)["code"], "bad-arg")
        self.assertEqual(b.calls, [])

    def test_rate_limit(self):
        h, b, _ = mk(rate={**rh.RATE, "vm-sysctl": (3, 60)})
        a = {"key": "vm.swappiness", "value": 10}
        codes = [h.handle(CREDS, req("vm-sysctl", a))["code"] for _ in range(5)]
        self.assertEqual(codes, ["ok", "ok", "ok", "rate-limited", "rate-limited"])
        h._t[0] += 61
        self.assertEqual(h.handle(CREDS, req("vm-sysctl", a))["code"], "ok")
        self.assertEqual(len(b.calls), 4)

    def test_rate_limit_is_per_verb(self):
        h, b, _ = mk(rate={**rh.RATE, "vm-sysctl": (1, 60)})
        a = {"key": "vm.swappiness", "value": 10}
        h.handle(CREDS, req("vm-sysctl", a))
        self.assertEqual(h.handle(CREDS, req("vm-sysctl", a))["code"], "rate-limited")
        self.assertEqual(h.handle(CREDS, req("gpu-power", {"watts": 200}))["code"], "ok")

    def test_garbage_does_not_use_rate_budget(self):
        h, b, _ = mk(rate={**rh.RATE, "vm-sysctl": (1, 60)})
        for _ in range(5):
            h.handle(CREDS, req("vm-sysctl", {"key": "nope", "value": 1}))
        self.assertEqual(h.handle(CREDS, req("vm-sysctl", {"key": "vm.swappiness", "value": 10}))["code"], "ok")

    def test_drop_caches_once_per_60s(self):
        h, b, _ = mk()
        self.assertEqual(h.handle(CREDS, req("drop-caches", {"level": 1}))["code"], "ok")
        self.assertEqual(h.handle(CREDS, req("drop-caches", {"level": 3}))["code"], "precondition")
        h._t[0] += 61
        self.assertEqual(h.handle(CREDS, req("drop-caches", {"level": 3}))["code"], "ok")

    def test_audit_lines(self):
        h, b, lines = mk()
        h.handle(CREDS, req("gpu-power", {"watts": 200}, rid="ok1"))
        h.handle(CREDS, req("nope", {}, rid="bad1"))
        h.handle(CREDS, req("gpu-power", {"watts": 5000}, rid="bad2"))
        h.handle(CREDS, b"junk")
        rows = [json.loads(x) for x in lines]
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["code"] for r in rows], ["ok", "bad-verb", "bad-arg", "bad-arg"])
        self.assertEqual(rows[0]["verb"], "gpu-power")
        self.assertEqual(rows[0]["args"], {"watts": 200})
        self.assertEqual(rows[0]["peer_uid"], ME)
        self.assertEqual(rows[0]["id"], "ok1")
        for x in lines:
            self.assertNotIn("\n", x)

    def test_preconditions(self):
        h, b, _ = mk()
        b.in_use = True
        self.assertEqual(h.handle(CREDS, req("module-unload", {"names": ["nvidia"]}))["code"], "precondition")
        b.in_use = False
        self.assertEqual(h.handle(CREDS, req("module-unload", {"names": ["nvidia_drm", "nvidia"]}))["code"], "ok")
        self.assertEqual(h.handle(CREDS, req("pcie-maxreadreq", {"bdf": "0000:02:00.0", "value": 512}))["code"], "precondition")
        self.assertEqual(h.handle(CREDS, req("pcie-maxreadreq", {"bdf": "0000:01:00.0", "value": 512}))["code"], "ok")
        self.assertEqual(h.handle(CREDS, req("gpu-clocks", {"min_mhz": 100, "max_mhz": 2000}))["code"], "precondition")
        self.assertEqual(h.handle(CREDS, req("gpu-clocks", {"min_mhz": 300, "max_mhz": 2000}))["code"], "ok")
        self.assertEqual(h.handle(CREDS, req("gpu-power", {"watts": 99}))["code"], "precondition")
        self.assertEqual(h.handle(CREDS, req("cpu-governor", {"governor": "schedutil", "epp": "power"}))["code"], "precondition")
        self.assertEqual(h.handle(CREDS, req("cpu-governor", {"governor": "powersave", "epp": "power"}))["code"], "ok")
        b.profile_ok = False
        self.assertEqual(h.handle(CREDS, req("apparmor-reload"))["code"], "precondition")

    def test_backend_failure_is_failed_not_crash(self):
        h, b, _ = mk()
        b.fail = True
        self.assertEqual(h.handle(CREDS, req("module-load", {"name": "nvidia"}))["code"], "failed")

    def test_pure_helpers(self):
        self.assertEqual(rh.mrrs_new_value(0x2810, 4096), 0x5810)
        self.assertTrue(rh.peer_allowed((1, 5, 5), {5}))
        self.assertFalse(rh.peer_allowed((1, 6, 5), {5}))
        self.assertFalse(rh.peer_allowed(None, {5}))

    def test_fuzz_never_executes_unknown(self):
        rnd = random.Random(7)
        h, b, _ = mk(rate={v: (10 ** 9, 1) for v in rh.RATE})
        alphabet = string.printable + "ñ‮\x00"
        for _ in range(3000):
            s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 80)))
            verb = rnd.choice(list(rh.SCHEMAS) + [s])
            arg = rnd.choice([s, [s], rnd.randint(-10 ** 12, 10 ** 12), None, {"x": s}])
            keys = list(rh.SCHEMAS.get(verb, {"a": 0})) or ["a"]
            h.handle(CREDS, json.dumps({"verb": verb, "args": {k: arg for k in keys}, "id": "f"}).encode())
        allowed_mod = set(rh.LOADABLE)
        for c in b.calls:
            if c[0] == "module_load":
                self.assertIn(c[1], allowed_mod)
            if c[0] == "module_unload":
                self.assertTrue(set(c[1]) <= set(rh.NVIDIA_UNLOAD_ORDER))
            if c[0] == "vm":
                self.assertIn(c[1], rh.VM_KEYS)


class SystemBackendArgv(unittest.TestCase):
    """Real SystemBackend with the process runner stubbed: argv lists only."""
    def setUp(self):
        self.seen = []
        def runner(argv, timeout):
            self.seen.append((argv, timeout))
            return 0, "2810\n", ""
        self.sb = rh.SystemBackend(runner)

    def test_argv_lists_absolute_strings(self):
        self.sb.module_load("nvidia")
        self.sb.module_unload(["nvidia_drm", "nvidia"])
        try:
            self.sb.pcie_maxreadreq("0000:01:00.0", 4096)
        except rh.BackendError:
            pass  # stub returns the same value on read back
        self.sb.gpu_clocks(300, 2000)
        self.sb.gpu_power(200)
        self.assertTrue(self.seen)
        for argv, timeout in self.seen:
            self.assertIsInstance(argv, list)
            self.assertTrue(argv[0].startswith("/"), argv)
            self.assertTrue(all(isinstance(a, str) for a in argv))
            self.assertNotIn("-f", argv)
            self.assertNotIn("--force", argv)
        self.assertEqual(self.seen[0][0][1:], ["--", "nvidia"])
        self.assertEqual(self.seen[1][0][1:], ["--", "nvidia_drm", "nvidia"])

    def test_shell_metachar_cannot_reach_argv(self):
        h = rh.Helper(self.sb, {ME}, audit=lambda s: None)
        h.handle(CREDS, req("module-load", {"name": "nvidia; reboot"}))
        h.handle(CREDS, req("module-load", {"name": "$(id)"}))
        self.assertEqual(self.seen, [])


class StaticGate(unittest.TestCase):
    FILES = [ROOT / "gb_root_helper.py", ROOT / "gb_helper_client.py"]

    def test_no_shell_no_os_system_stdlib_only(self):
        for f in self.FILES:
            tree = ast.parse(f.read_text())
            for n in ast.walk(tree):
                if isinstance(n, ast.keyword) and n.arg == "shell":
                    self.assertIsInstance(n.value, ast.Constant, f)
                    self.assertIs(n.value.value, False, f"{f.name}: shell must be False")
                if isinstance(n, ast.Attribute) and n.attr in ("system", "popen", "spawnl", "spawnlp", "spawnv", "execv", "execl"):
                    if isinstance(n.value, ast.Name) and n.value.id == "os":
                        self.fail(f"{f.name}: os.{n.attr}")
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
                    for m in mods:
                        top = m.split(".")[0]
                        self.assertIn(top, sys.stdlib_module_names, f"{f.name} imports {m}")
            self.assertNotIn("shell=True", f.read_text())

    def test_single_subprocess_call_site_with_variable_argv(self):
        tree = ast.parse(self.FILES[0].read_text())
        sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess" and n.func.attr != "TimeoutExpired"]
        self.assertEqual(len(sites), 1)
        self.assertIsInstance(sites[0].args[0], ast.Name)  # argv variable, never a string
        self.assertEqual(sites[0].args[0].id, "argv")

    def test_helper_is_small(self):
        code = [x for x in self.FILES[0].read_text().splitlines() if x.strip() and not x.strip().startswith("#")]
        self.assertLess(len(code), 500)  # design goal 400, actual about 450: see README


class RealSocket(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(dir=CACHE)
        self.path = os.path.join(self.td.name, "h.sock")

    def tearDown(self):
        self.td.cleanup()

    def start(self, uids):
        h, b, lines = mk(uids=uids)
        lst = rh.listener_from_env_or_path(self.path)
        th = threading.Thread(target=h.serve_forever, args=(lst,), daemon=True)
        th.start()
        def stop():
            h.stop = True
            th.join(3)
            lst.close()
        self.addCleanup(stop)
        return h, b, lines

    def test_peercred_matches_and_roundtrip(self):
        h, b, lines = self.start({ME})
        r = client.HelperClient(self.path, timeout=5).drop_caches(1)
        self.assertEqual((r["ok"], r["code"]), (True, "ok"))
        self.assertEqual(b.calls, [("drop", 1)])
        row = json.loads(lines[-1])
        self.assertEqual(row["peer_uid"], ME)
        self.assertEqual(row["peer_pid"], os.getpid())
        a, c = socket.socketpair()
        self.assertEqual(rh.Helper.peercred(a)[1], ME)
        a.close(); c.close()

    def test_wrong_uid_over_socket(self):
        h, b, _ = self.start({ME + 1})
        r = client.HelperClient(self.path, timeout=5).module_load("nvidia")
        self.assertEqual(r["code"], "peer-denied")
        self.assertEqual(b.calls, [])

    def test_oversized_frame_over_socket(self):
        h, b, _ = self.start({ME})
        s = socket.socket(socket.AF_UNIX); s.settimeout(5); s.connect(self.path)
        s.sendall(struct.pack(">I", 10 ** 6))
        (n,) = struct.unpack(">I", s.recv(4))
        self.assertEqual(json.loads(s.recv(n))["code"], "bad-arg")
        s.close()
        self.assertEqual(b.calls, [])

    def test_two_requests_one_connection_second_ignored(self):
        h, b, _ = self.start({ME})
        s = socket.socket(socket.AF_UNIX); s.settimeout(5); s.connect(self.path)
        body = req("gpu-power", {"watts": 200})
        frame = struct.pack(">I", len(body)) + body
        s.sendall(frame + frame)
        (n,) = struct.unpack(">I", s.recv(4)); s.recv(n)
        try:
            self.assertEqual(s.recv(1), b"")  # closed after one reply
        except ConnectionResetError:
            pass  # unread second frame makes the close a reset: also closed
        s.close()
        self.assertEqual(len(b.calls), 1)

    def test_client_without_helper_does_not_fall_back(self):
        r = client.HelperClient(os.path.join(self.td.name, "none.sock"), timeout=1).module_unload(["nvidia"])
        self.assertEqual((r["ok"], r["code"], r["detail"]), (False, "failed", "helper-unavailable"))


if __name__ == "__main__":
    unittest.main()
