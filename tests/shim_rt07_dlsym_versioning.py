#!/usr/bin/env python3
"""RT-07 non-device test driver.  No GPU, no driver, no root.

Builds a stub libcuda.so.1 and two probes (dlsym/dlopen bound at GLIBC_2.34 and GLIBC_2.2.5)
in a scratch dir, then runs each probe (a) with the shim under test preloaded and (b) without it
(control).  With the shim from before the RT-07 patch the 2.34 probe FAILS; after it, all pass.

usage: shim_rt07_dlsym_versioning.py /path/to/libgreenboost_cuda.so [scratch_dir]
env GB_PRELOAD_PREFIX: extra library preloaded before the shim (libasan.so for a sanitizer build).
"""
import os, subprocess, sys, tempfile

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 2
    shim = os.path.realpath(sys.argv[1])
    here = os.path.dirname(os.path.realpath(__file__))
    src = os.path.join(here, "shim_rt07_dlsym_versioning.c")
    out = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="gb_rt07_")
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    cc = ["gcc", "-Wall", "-Wextra", "-O1", "-g"]
    stub = os.path.join(out, "libcuda.so.1")
    subprocess.run(cc + ["-shared", "-fPIC", "-DGB_STUB", "-Wl,-soname,libcuda.so.1", "-o", stub, src], check=True)
    probes = {"2.34": [], "2.2.5": ["-DGB_BIND_225"]}
    env_base = dict(os.environ, LD_LIBRARY_PATH=out + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
                    GREENBOOST_ACTIVE="1")
    bad = 0
    for tag, flags in probes.items():
        exe = os.path.join(out, "probe_" + tag.replace(".", "_"))
        subprocess.run(cc + flags + [src, "-ldl", "-o", exe], check=True)
        for mode in ("control", "preload"):
            env = dict(env_base)
            if mode == "preload":
                env["LD_PRELOAD"] = (os.environ.get("GB_PRELOAD_PREFIX", "") + " " + shim).strip()
                env.setdefault("ASAN_OPTIONS", "detect_leaks=0")
                env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1")
            r = subprocess.run([exe, os.path.basename(shim), mode] + (["prefix"] if os.environ.get("GB_PRELOAD_PREFIX") else []), env=env, capture_output=True, text=True, timeout=60)
            print(f"--- probe bound at {tag}, {mode} (exit {r.returncode})")
            print(r.stdout.rstrip())
            if r.returncode != 0:
                bad += 1
    print("TOTAL:", "FAIL" if bad else "OK")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
