#!/usr/bin/env python3
"""RT-13 (shim T2 cap overshoot) non-device test driver.  No GPU, no root.

1. Extracts the GB_T2_CAP_BEGIN..END region from the shim source and compiles it into
   shim_rt07_rt13_pool_cap.c (fake allocator, 16 threads), with -Wall -Wextra, plain and with
   ASan+UBSan.
2. Static check on the shim source: no check-then-add of gb_t2_overflow_bytes is left in the
   Path A / Path B / cuMemCreate host-fallback code of gb_overflow_alloc_ex / cuMemCreate.

usage: shim_rt07_rt13_pool_cap.py /path/to/greenboost_cuda_shim.c [scratch_dir]
"""
import os, re, subprocess, sys, tempfile

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 2
    src = os.path.realpath(sys.argv[1])
    here = os.path.dirname(os.path.realpath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="gb_rt13_")
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    text = open(src, errors="replace").read()
    m = re.search(r"/\* GB_T2_CAP_BEGIN.*?GB_T2_CAP_END \*/", text, re.S)
    if not m:
        print("FAIL: GB_T2_CAP_BEGIN/END markers not found (patch shim-0005 not applied?)"); return 1
    pct = re.search(r"#define\s+GB_T2_INFERENCE_CAP_PCT\s+(\d+)", text)
    region = os.path.join(out, "gb_t2_cap_region.h")
    open(region, "w").write(m.group(0) + "\n")
    test_c = os.path.join(here, "shim_rt07_rt13_pool_cap.c")
    base = ["gcc", "-Wall", "-Wextra", "-std=gnu11", "-g", f'-DGB_CAP_REGION="{region}"',
            f"-DGB_CAP_PCT={pct.group(1) if pct else 88}", test_c, "-lpthread"]
    bad = 0
    for tag, flags in (("plain-O2", ["-O2"]),
                       ("asan+ubsan", ["-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])):
        exe = os.path.join(out, "t_" + tag.replace("+", "_"))
        r = subprocess.run(base + flags + ["-o", exe], capture_output=True, text=True)
        if r.returncode or r.stderr.strip():
            print(f"--- {tag}: build output:\n{r.stderr}")
        if r.returncode:
            print(f"FAIL {tag} (build failed)"); bad += 1; continue
        r = subprocess.run([exe], capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, ASAN_OPTIONS="detect_leaks=0", UBSAN_OPTIONS="halt_on_error=1"))
        print(f"--- {tag} (exit {r.returncode})\n{r.stdout.rstrip()}\n{r.stderr.rstrip()[:800]}")
        bad += r.returncode != 0
    # static: the old tail charge must be gone from the three host-T2 sites
    adds = [i + 1 for i, l in enumerate(text.splitlines())
            if re.search(r"atomic_fetch_add_explicit\(&gb_t2_overflow_bytes,\s*(alloc_bytesize|size)\b", l)]
    print("static: remaining charge-after-alloc sites (alloc_bytesize/size):", adds or "none")
    # the line 1802-style 'aligned - size' delta charge is a different, intentional site and is not matched
    bad += bool(adds)
    print("TOTAL:", "FAIL" if bad else "OK")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
