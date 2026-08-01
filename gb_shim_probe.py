#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_shim_probe.py — evidence-based, cached verdict for "does the GreenBoost
CUDA shim (LD_PRELOAD=libgreenboost_cuda.so) actually work with the currently
installed llama.cpp engine on this box?"

Why this exists: `gb_synapse_backends.py` used to hardcode the shim OFF for
llama.cpp behind a comment blaming a specific failure — `cudaFuncGetAttributes`
returning "invalid device function" inside ggml's `ggml_cuda_kernel_can_use_pdl`
(see `greenboost_cuda_shim.c`'s `gb_cudart_rebind()` for the root cause: a
cudart-major mismatch between the shim's own dlopen'ed fallback and the one
ggml's CUDA backend late-dlopens). That failure mode was real when the comment
was written, but the shim has since grown a fix for exactly this case
(canonical-`realpath` cudart comparison + a per-fatbin rebind re-arm in
`__cudaRegisterFunction`), and the hardcoded off-switch was never revisited —
the comment's own words were "Probe once, then decide — never assume," and no
probe existed. This module is that probe.

The verdict is cached (`_verdict_path()`) keyed on everything that could make
a stale verdict wrong: the shim `.so`'s mtime, the installed engine version,
and the ggml CUDA backend's own cudart soname. Any of those changing (a shim
rebuild, an engine rebuild, a driver/CUDA upgrade that swaps the linked
cudart) invalidates the cache automatically — this is the mechanism that was
missing and let the original verdict go stale for two weeks undetected.

`GB_SYNAPSE_SHIM=1`/`=0` still force the decision (fast path for anyone who
already knows their answer, and an escape hatch if the probe itself is
untrustworthy on some box); unset means "ask the probe."
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))

_VERDICT_PATH = Path(
    os.environ.get("GB_SHIM_PROBE_CACHE",
                    "/var/lib/greenboost/synapse/shim_probe.json"))
_PROBE_TIMEOUT_S = 60
_FAIL_MARKERS = (
    "invalid device function",
    "cuda error",
    "cuda_error",
    "illegal memory access",
)


def _shim_path() -> "Path | None":
    try:
        import gb_cluster
        p = Path(gb_cluster.GREENBOOST_SHIM)
    except Exception:
        p = Path("/usr/local/lib/libgreenboost_cuda.so")
    return p if p.is_file() else None


def _ggml_cudart_soname(engine_dir: Path) -> str:
    """The cudart soname ggml's CUDA backend actually links against (e.g.
    'libcudart.so.12') — the other half of the mismatch the shim's rebind fix
    targets. Read via `ldd`; best-effort, empty string if undeterminable."""
    lib = engine_dir / "libggml-cuda.so"
    if not lib.is_file():
        return ""
    try:
        out = subprocess.run(["ldd", str(lib)], capture_output=True,
                              text=True, timeout=10).stdout
    except Exception:
        return ""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("libcudart.so"):
            return line.split(" ", 1)[0]
    return ""


def _cache_key(engine_dir: Path) -> dict:
    shim = _shim_path()
    engine_version = ""
    try:
        import gb_synapse
        engine_version = gb_synapse.engine_version()
    except Exception:
        pass
    return {
        "shim_mtime": shim.stat().st_mtime if shim else None,
        "shim_path": str(shim) if shim else None,
        "engine_version": engine_version,
        "ggml_cudart_soname": _ggml_cudart_soname(engine_dir),
    }


def _smallest_local_gguf(engine_dir: Path) -> "Path | None":
    """The smallest locally cached GGUF, used to run a real (not synthetic)
    llama-server load through the shim — reproducing ggml's late dlopen of
    its own cudart is only possible by actually loading the real engine, a
    plain cudart-only test binary does not reproduce it (see module
    docstring). Falls back to None (probe skipped) if nothing is cached."""
    try:
        import gb_synapse
        candidates = [
            (e.n_bytes, Path(e.path))
            for e in gb_synapse.list_models()
            if e.engine == "llama.cpp" and Path(e.path).is_file()
        ]
    except Exception:
        candidates = []
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0] or 1 << 62)
    return candidates[0][1]


def _run_probe(engine_dir: Path, shim: Path) -> tuple[bool, str]:
    model = _smallest_local_gguf(engine_dir)
    if model is None:
        return False, "no local GGUF to probe with — skipped (treat as untested, not failed)"

    port = 18122  # fixed internal-only probe port, never proxied
    cmd = [str(engine_dir / "llama-server"), "-m", str(model),
           "--host", "127.0.0.1", "--port", str(port),
           "--ctx-size", "256", "-ngl", "1", "-n", "1",
           "--no-webui", "-fit", "off"]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(shim)
    env["GREENBOOST_ACTIVE"] = "1"
    env["LD_LIBRARY_PATH"] = str(engine_dir) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

    import tempfile
    import urllib.error
    import urllib.request

    # llama-server is a long-lived HTTP daemon — it never exits on its own,
    # so "ran for the full timeout" is not ambiguous, it IS the success
    # signal, and the only reliable way to tell that apart from "still
    # starting up" or "hung" is to actually poll /health while watching for
    # early death. subprocess.run(timeout=...)'s TimeoutExpired can't
    # distinguish "healthy and still serving" from "silently hung" — this
    # loop can, by hitting the port. Real crashes here (registration
    # failure, VMM alloc abort — both confirmed live 2026-07-31/08-01) exit
    # the process almost immediately (well under a second in both cases),
    # so poll() catching an early exit is the actual failure signal, not a
    # last-resort marker scan.
    with tempfile.TemporaryFile(mode="w+") as logf:
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                 text=True)
        try:
            deadline = time.time() + _PROBE_TIMEOUT_S
            healthy = False
            while time.time() < deadline:
                ret = proc.poll()
                if ret is not None:
                    break
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            healthy = True
                            break
                except (urllib.error.URLError, OSError, TimeoutError):
                    pass
                time.sleep(1.0)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

        logf.seek(0)
        out = logf.read().lower()

    if healthy:
        return True, "probe loaded the real engine under the shim and answered /health with no CUDA/device errors"
    for marker in _FAIL_MARKERS:
        if marker in out:
            return False, f"reproduced known failure ({marker!r} in probe output)"
    if proc.returncode is not None and proc.returncode not in (0, -9, -15):
        return False, f"probe exited {proc.returncode} before becoming healthy: {out[-400:]}"
    return False, "probe never answered /health within timeout — treating as unsupported"


def shim_works_for_llama(engine_dir: "Path | None" = None) -> tuple[bool, str]:
    """(works, reason) — the single decision point `gb_synapse_backends.py`
    consults before deciding whether to LD_PRELOAD the shim into a
    llama-server child. Cached; only re-probes when the cache key
    (shim build, engine build, ggml's cudart soname) changes."""
    forced = os.environ.get("GB_SYNAPSE_SHIM")
    if forced == "1":
        return True, "forced on (GB_SYNAPSE_SHIM=1)"
    if forced == "0":
        return False, "forced off (GB_SYNAPSE_SHIM=0)"

    shim = _shim_path()
    if shim is None:
        return False, "shim not installed"

    if engine_dir is None:
        try:
            import gb_synapse
            engine_dir = gb_synapse.ENGINE_DIR
        except Exception:
            return False, "engine dir unresolvable"

    key = _cache_key(engine_dir)
    cached = _read_cache()
    if cached is not None and cached.get("key") == key:
        return bool(cached["works"]), f"cached: {cached['reason']}"

    works, reason = _run_probe(engine_dir, shim)
    _write_cache(key, works, reason)
    return works, reason


def _read_cache() -> "dict | None":
    try:
        return json.loads(_VERDICT_PATH.read_text())
    except Exception:
        return None


def _write_cache(key: dict, works: bool, reason: str) -> None:
    try:
        _VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _VERDICT_PATH.write_text(json.dumps({
            "key": key, "works": works, "reason": reason, "ts": time.time(),
        }, indent=2))
    except Exception:
        pass  # best-effort cache; a failed write just means re-probing next time


if __name__ == "__main__":
    ok, why = shim_works_for_llama()
    print(json.dumps({"shim_works_for_llama": ok, "reason": why}, indent=2))
    sys.exit(0 if ok else 1)
