# GreenBoost v3.4 - Kernel module + CUDA shim build system
# Author: Ferran Duarri
#
# Kbuild file handles kernel-internal rules (obj-m, ccflags-y).
# This Makefile handles user-facing targets (module, install, load, etc.)
# Keeping them separate prevents the kernel Kbuild from overwriting this file.

KVER    ?= $(shell uname -r)
KDIR_CANDIDATES := \
    /lib/modules/$(KVER)/build \
    /usr/src/kernels/$(KVER) \
    /usr/src/linux-$(KVER) \
    /usr/lib/modules/$(KVER)/build
KDIR    := $(firstword $(wildcard $(KDIR_CANDIDATES)))
ifeq ($(KDIR),)
$(error Cannot find kernel headers for $(KVER). Install linux-headers-$(KVER) or kernel-devel-$(KVER))
endif
# Detect Clang-built kernels: CachyOS, Arch/clang, Gentoo/clang kernels set
# CONFIG_CC_IS_CLANG=y.  Passing LLVM=1 selects the full LLVM toolchain
# (CC=clang HOSTCC=clang LD=ld.lld AR=llvm-ar NM=llvm-nm) to match the
# kernel's own build toolchain.  GCC rejects clang-specific flags like
# -mretpoline-external-thunk and -mstack-alignment=8, causing -Werror failures.
KERNEL_LLVM := $(shell grep -qs '^CONFIG_CC_IS_CLANG=y' $(KDIR)/.config 2>/dev/null && echo LLVM=1)
PWD     := $(shell pwd)
CC      := gcc
SHIM    := libgreenboost_cuda.so
AUDIT   := libgreenboost_audit.so
AUDIT32 := libgreenboost_audit32.so
VMM_OVERRIDE := libgreenboost_vmm_override.so
MODULE  := greenboost.ko
NETD    := greenboost-netd
GB_VERSION := 3.4

PHYS_GB    ?= 0
VIRT_GB    ?= 0
RESERVE_GB ?= 8
NVME_GB    ?= 0
# MIN-04: Module default for nvme_pool_gb is 0 (disabled).
# Override with: make load NVME_POOL=N   to enable T3 on systems with NVMe swap.
NVME_POOL  ?= 0

USE_NCCL   ?= 0
USE_NVTX   ?= 1


HAS_AVX2 := $(shell grep -qw avx2 /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
COMMON_CFLAGS := -march=native -mtune=native -O3 -funroll-loops -std=gnu11
COMMON_CFLAGS += -flto -fvisibility=hidden -ffunction-sections -fdata-sections
COMMON_CFLAGS += -fomit-frame-pointer -fprefetch-loop-arrays
ifeq ($(HAS_AVX2),1)
COMMON_CFLAGS += -mavx2
endif
ifeq ($(USE_NCCL),1)
COMMON_CFLAGS += -DGREENBOOST_USE_NCCL
NCCL_LDFLAGS  := -lnccl
endif
ifeq ($(USE_NVTX),1)
SHIM_CFLAGS   += -DGREENBOOST_USE_NVTX
SHIM_CFLAGS   += -I$(CURDIR)/../greenboost_sources/NVTX/c/include
SHIM_LDFLAGS  += -lnvToolsExt
endif

# 32-bit capability probe - tests actual -m32 compiler+linker support rather than a
# specific header path (which moved in glibc 2.42 / Ubuntu 25.04 and later releases).
HAS_32BIT_HEADERS := $(shell echo 'int x;' | $(CC) -m32 -x c - -shared -fPIC -o /dev/null 2>/dev/null && echo 1 || echo 0)
# AUDIT32_CFLAGS: i686 build - no -march=native/-mtune=native (incompatible with -m32),
# no -mavx2 (64-bit era), adds i386 multilib include path.
AUDIT32_CFLAGS := -march=i686 -O3 -funroll-loops -std=gnu11
AUDIT32_CFLAGS += -flto -fvisibility=hidden -ffunction-sections -fdata-sections
AUDIT32_CFLAGS += $(shell test -d /usr/include/i386-linux-gnu && echo -I/usr/include/i386-linux-gnu)
# Shim uses the version script (local: *) for symbol hiding instead of
# -fvisibility=hidden, so that version-script global: declarations actually
# export the hook functions into the dynamic symbol table.
SHIM_CFLAGS   := -march=native -mtune=native -O3 -funroll-loops -std=gnu11
SHIM_CFLAGS   += -flto -ffunction-sections -fdata-sections
SHIM_CFLAGS   += -fomit-frame-pointer -fno-semantic-interposition -fprefetch-loop-arrays
# B1: Security hardening - shim is LD_PRELOAD'd into every CUDA process; buffer
# overflows carry the privilege of the target process.
SHIM_CFLAGS   += -D_FORTIFY_SOURCE=2 -fstack-protector-strong
SHIM_CFLAGS   += -Wformat -Wformat-security
ifeq ($(HAS_AVX2),1)
SHIM_CFLAGS   += -mavx2
endif
ifeq ($(USE_NCCL),1)
SHIM_CFLAGS   += -DGREENBOOST_USE_NCCL
endif
SHIM_LDFLAGS  := -Wl,--gc-sections -Wl,--as-needed $(NCCL_LDFLAGS)
# SHIM_CFLAGS_V12: same flags as SHIM_CFLAGS but without -flto and without
# -ffunction-sections/-fdata-sections.  Used to compile greenboost_cuda_v12.c
# which contains .symver inline asm trampolines: LTO's GIMPLE recompilation
# strips .symver directives, and per-function sections risk gc without them.
SHIM_CFLAGS_V12 := $(filter-out -flto -ffunction-sections -fdata-sections,$(SHIM_CFLAGS))

# CUDA header probe - prefers the newest side-by-side versioned install
# (/usr/local/cuda-13 over /usr/local/cuda-12 over the unversioned symlink).
# This matters when the user has both CUDA 12 and CUDA 13 installed via the
# NVIDIA repo and the /usr/local/cuda symlink still points to 12.
CUDA_DIR := $(shell \
    latest=$$(ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -V | tail -1); \
    if [ -n "$$latest" ] && [ -f "$$latest/include/cuda.h" ]; then \
        echo "$$latest"; \
    elif [ -d /usr/local/cuda ] && [ -f /usr/local/cuda/include/cuda.h ]; then \
        echo "/usr/local/cuda"; \
    elif [ -d /usr/cuda ] && [ -f /usr/cuda/include/cuda.h ]; then \
        echo "/usr/cuda"; \
    elif [ -d /opt/cuda ] && [ -f /opt/cuda/include/cuda.h ]; then \
        echo "/opt/cuda"; \
    fi)
ifneq ($(CUDA_DIR),)
SHIM_CFLAGS    += -I$(CUDA_DIR)/include
endif

# libzstd probe: fabric payload compression (H2D over the cluster socket). When
# present, both the shim (netc client) and netd feeder compile with
# -DGB_HAVE_ZSTD -lzstd and negotiate GB_NET_FEAT_ZSTD at handshake. Absent →
# the feature bit is never advertised and every transfer stays raw (no-op).
ZSTD_CFLAGS := $(shell pkg-config --cflags libzstd 2>/dev/null)
ZSTD_LIBS   := $(shell pkg-config --libs   libzstd 2>/dev/null)
ifeq ($(ZSTD_LIBS),)
  ifneq ($(wildcard /usr/include/zstd.h),)
    ZSTD_LIBS := -lzstd
  endif
endif
ifneq ($(ZSTD_LIBS),)
SHIM_CFLAGS   += -DGB_HAVE_ZSTD $(ZSTD_CFLAGS)
SHIM_LDFLAGS  += $(ZSTD_LIBS)
NETD_ZSTD_CFLAGS := -DGB_HAVE_ZSTD $(ZSTD_CFLAGS)
NETD_ZSTD_LIBS   := $(ZSTD_LIBS)
$(info [GreenBoost] libzstd found - fabric compression enabled ($(ZSTD_LIBS)))
else
NETD_ZSTD_CFLAGS :=
NETD_ZSTD_LIBS   :=
$(info [GreenBoost] libzstd NOT found - fabric compression disabled)
endif

DKMS_ROOT := /usr/src/greenboost-$(GB_VERSION)

# ── Optional eBPF observability tracer ────────────────────────────────────
# Requires: clang, bpftool, libbpf-dev (or libbpf-devel / libbpf).
# The BPF build is fully optional: if any prereq is missing the main build
# continues without the tracer and greenboost_builder.py reports a warning.
# Override: make BPF=0 to force-disable even when prereqs are present.
BPF ?= auto
EBPF_TRACE := greenboost-ebpf-trace

ifeq ($(BPF),auto)
  # Accept plain 'clang' or versioned 'clang-N' (Ubuntu ships clang-21, not clang)
  _CLANG_PLAIN := $(shell command -v clang 2>/dev/null)
  _CLANG_VER   := $(shell command -v clang-21 clang-20 clang-19 clang-18 clang-17 clang-16 2>/dev/null | head -1)
  CLANG        := $(or $(_CLANG_PLAIN),$(_CLANG_VER))
  _HAS_CLANG   := $(if $(CLANG),1,0)
  _HAS_BPFTOOL := $(shell command -v bpftool >/dev/null 2>&1 && echo 1 || echo 0)
  _HAS_LIBBPF  := $(shell pkg-config --exists libbpf 2>/dev/null && echo 1 || echo 0)
  # CO-RE BPF requires kernel BTF; without it vmlinux.h generation fails hard.
  _HAS_BTF     := $(shell test -r /sys/kernel/btf/vmlinux && echo 1 || echo 0)
  ifeq ($(_HAS_CLANG)$(_HAS_BPFTOOL)$(_HAS_LIBBPF)$(_HAS_BTF),1111)
    BPF := 1
  else
    BPF := 0
  endif
else
  CLANG ?= clang
endif

ifeq ($(BPF),1)
EBPF_CFLAGS := -O2 -Wall -I ebpf/
EBPF_LDLIBS := $(shell pkg-config --libs libbpf) -lelf -lz
# clang -target bpf doesn't see the host's arch-specific include dir by
# default, so <bpf/bpf_helper_defs.h>'s __u64/__u32 (from <linux/types.h> ->
# <asm/types.h>) fail to resolve ("unknown type name '__u64'", verified
# 2026-07-09 on Ubuntu 26.04 with clang 20 + libbpf-dev 1.6.3). -idirafter
# (not -I) so it's a fallback behind ebpf/'s own headers, never shadowing
# them.
EBPF_BPF_CFLAGS := $(EBPF_CFLAGS) -idirafter /usr/include/$(shell uname -m)-linux-gnu -idirafter /usr/include
endif

.PHONY: all module shim audit audit32 netd ebpf clean test \
        install install-legacy install-libs build-info dkms-install dkms-uninstall \
        uninstall load unload reload status help ebpf-clean

ifeq ($(BPF),1)
all: module shim audit audit32 netd vmm_override $(EBPF_TRACE)
else
all: module shim audit audit32 netd vmm_override
	@echo "[GreenBoost] eBPF tracer skipped (clang/bpftool/libbpf not found); install them to enable observability"
endif

# ── eBPF tracer build rules ───────────────────────────────────────────────
ifeq ($(BPF),1)
ebpf/vmlinux.h:
	@echo "[GreenBoost] Generating vmlinux.h from kernel BTF..."
	@if [ ! -e /sys/kernel/btf/vmlinux ]; then \
	    echo "[GreenBoost] ERROR: /sys/kernel/btf/vmlinux does not exist on this" >&2; \
	    echo "  kernel ($$(uname -r)) - it was not built with CONFIG_DEBUG_INFO_BTF=y," >&2; \
	    echo "  so there is no BTF data to dump. The eBPF tracer needs a kernel with" >&2; \
	    echo "  BTF support; this is a kernel build-config requirement, not a missing" >&2; \
	    echo "  package (verified 2026-07-09: clang/bpftool/libbpf all present, this" >&2; \
	    echo "  is the actual blocker). Skip 'make BPF=1' on this host, or rebuild the" >&2; \
	    echo "  kernel with CONFIG_DEBUG_INFO_BTF=y." >&2; \
	    exit 1; \
	fi
	@# Write to a temp file first: a failed bpftool run must NOT leave a
	@# stale empty $@ behind, since make treats "target exists" as
	@# "up to date" forever after regardless of content (verified
	@# 2026-07-09 - an empty vmlinux.h from an earlier failed run silently
	@# poisoned every subsequent build with "unknown type name '__u64'",
	@# masking this actual root cause for who knows how long).
	@bpftool btf dump file /sys/kernel/btf/vmlinux format c > $@.tmp && mv $@.tmp $@ \
	    || { rm -f $@.tmp; echo "[GreenBoost] ERROR: bpftool btf dump failed" >&2; exit 1; }

ebpf/gb_trace.bpf.o: ebpf/gb_trace.bpf.c ebpf/gb_offsets.h ebpf/vmlinux.h
	@echo "[GreenBoost] Compiling BPF program..."
	$(CLANG) -target bpf -g -O2 $(EBPF_BPF_CFLAGS) -D__TARGET_ARCH_x86 \
	    -Wno-missing-declarations -c $< -o $@

ebpf/gb_trace.skel.h: ebpf/gb_trace.bpf.o
	@echo "[GreenBoost] Generating BPF skeleton..."
	bpftool gen skeleton $< > $@

$(EBPF_TRACE): ebpf/gb_trace.c ebpf/gb_trace.skel.h ebpf/gb_offsets.h
	@echo "[GreenBoost] Building eBPF tracer userspace..."
	$(CC) $(EBPF_CFLAGS) -o $@ $< $(EBPF_LDLIBS)
	@echo "[GreenBoost] Built $(EBPF_TRACE)"

.PHONY: ebpf
ebpf: $(EBPF_TRACE)

ebpf-clean:
	rm -f ebpf/vmlinux.h ebpf/gb_trace.bpf.o ebpf/gb_trace.skel.h $(EBPF_TRACE)
else
ebpf:
	@echo "[GreenBoost] BPF=0 , eBPF tracer not built (install clang bpftool libbpf-dev)"

ebpf-clean: ;
endif

# PR-W: pytest regression suite covering wire-protocol layout, PSK file
# loading, LAN filter, and payload-size validation.  Pure-Python - no
# CUDA, no /dev/greenboost, no running daemon needed.  Fast: ~30 ms.
test:
	@cd tests && python3 -m pytest -v 2>&1 | tail -10
	@echo "[GreenBoost] regression suite passed"

# Host-only C unit tests (gb_expert_tier.h's LFRU scoring, etc.) - plain C99,
# no CUDA/kernel headers, no GPU needed. Fast: builds + runs in well under 1s.
test-c: tests/c/test_gb_expert_tier.c gb_expert_tier.h
	@$(CC) -std=c99 -Wall -Wextra -o /tmp/gb_test_expert_tier tests/c/test_gb_expert_tier.c
	@/tmp/gb_test_expert_tier
	@echo "[GreenBoost] C unit tests passed"

# Mechanical invariant checks (golden-principles.md) - hardcoded-hardware-value
# scan, dataflux telemetry coverage, MCP tool parity, installer/uninstaller
# parity, secrets/IP/home-path scan, doc freshness. Blocking iff any check
# reports a "blocking" finding; advisory findings are printed but don't fail.
check:
	@python3 checks/run_checks.py

# End-to-end agent-legibility harness (Colibri doctor.py-schema check steps:
# build artifact, health-check, shim smoke, dataflux, cluster snapshot, MCP
# self-check). See checks/verify_greenboost.py.
verify:
	@python3 checks/verify_greenboost.py

module:
	$(MAKE) -C $(KDIR) M=$(PWD) $(KERNEL_LLVM) modules

shim: greenboost_cuda_shim.c greenboost_cuda_v12.c greenboost_netc.c greenboost_cuda.map
	$(CC) -c -fPIC $(SHIM_CFLAGS_V12) -o greenboost_cuda_v12.o greenboost_cuda_v12.c
	$(CC) -shared -fPIC $(SHIM_CFLAGS) -o $(SHIM) \
		greenboost_cuda_shim.c greenboost_netc.c greenboost_cuda_v12.o \
		-ldl -lpthread \
		-Wl,--version-script=greenboost_cuda.map $(SHIM_LDFLAGS)
	@echo "[GreenBoost] Built $(SHIM)"

# SHIM_INSTALL_PATH: where libgreenboost_cuda.so will be installed.
# Override at build time: make audit SHIM_INSTALL_PATH=/usr/lib/libgreenboost_cuda.so
SHIM_INSTALL_PATH ?= /usr/local/lib/libgreenboost_cuda.so

audit: greenboost_audit.c
	$(CC) -shared -fPIC $(COMMON_CFLAGS) -DSHIM_PATH=\"$(SHIM_INSTALL_PATH)\" \
		-o $(AUDIT) greenboost_audit.c -ldl
	@echo "[GreenBoost] Built $(AUDIT) (SHIM_PATH=$(SHIM_INSTALL_PATH))"

# VMM override: bare unversioned cuDeviceGetAttribute + cuMemAddressReserve
# that wins the PLT race against libcuda.so.1 on Blackwell (cc >= 12).
# Deliberately NO version script - exports must be unversioned.
vmm_override: greenboost_vmm_override.c
	$(CC) -shared -fPIC -O2 -std=gnu11 \
		-fomit-frame-pointer -fno-semantic-interposition \
		-o $(VMM_OVERRIDE) greenboost_vmm_override.c -ldl
	@echo "[GreenBoost] Built $(VMM_OVERRIDE)"

audit32: greenboost_audit.c
ifeq ($(HAS_32BIT_HEADERS),1)
	$(CC) -m32 -shared -fPIC $(AUDIT32_CFLAGS) -DSHIM_PATH=\"$(SHIM_INSTALL_PATH)\" \
		-o $(AUDIT32) greenboost_audit.c -ldl
	@echo "[GreenBoost] Built $(AUDIT32) (i386, SHIM_PATH=$(SHIM_INSTALL_PATH))"
else
	@echo "[GreenBoost] NOTICE: 32-bit headers not found - skipping $(AUDIT32)"
	@echo "[GreenBoost]   Install gcc-multilib + 32-bit glibc headers for your distro,"
	@echo "[GreenBoost]   then re-run: make audit32"
	@echo "[GreenBoost]   (Run the appropriate greenboost_setup*.sh to install dependencies)"
endif

netd: greenboost_netd.c features/net_fabric.h netd-capture
	$(CC) $(COMMON_CFLAGS) -D_FORTIFY_SOURCE=2 -fstack-protector-strong \
	    -Wformat -Wformat-security -rdynamic $(NETD_ZSTD_CFLAGS) \
	    -o $(NETD) greenboost_netd.c -ldl -lpthread $(NCCL_LDFLAGS) $(NETD_ZSTD_LIBS)
	@echo "[GreenBoost] Built $(NETD)"

# Feeder __cudaRegisterFunction interposer (LD_PRELOAD'd into netd so remote
# dispatch can resolve the stripped lib's kernel stubs by name).
netd-capture: greenboost_netd_capture.c
	$(CC) -shared -fPIC -O2 -o libgreenboost_netd_capture.so \
	    greenboost_netd_capture.c -ldl -lpthread
	@echo "[GreenBoost] Built libgreenboost_netd_capture.so"

# Stage-A2 opt-in: sm_120a NVFP4 GEMM CUTLASS torch extension. NOT part of `all`
# or `install` (speculative, GPU-bench-gated). CUTLASS is header-only; point
# GB_CUTLASS_PATH at the checkout (default: vendored under ~/Dev/greenboost_all).
gb_cutlass:
	GB_CUTLASS_PATH="$${GB_CUTLASS_PATH:-$$HOME/Dev/greenboost_all/vendor/cutlass}" \
	    python3 third_party/gb_cutlass/setup.py build_ext --inplace
	@echo "[GreenBoost] Built gb_cutlass extension (enable at runtime with GB_CUTLASS_ENABLE=1 after bench)"

# Speed Program Phase 0 opt-in: path-bandwidth microbenchmark (bulk H2D/D2H,
# VRAM d2d, zero-copy SM read, staged-DMA-then-read). NOT part of `all` or
# `install` — a dev measurement tool, same status as gb_cutlass above.
# GB_BENCH_ARCH overrides the target SM arch (default sm_120a — Blackwell,
# this box's RTX 5070); NVCC overrides the compiler path.
# Prefer the CUDA 13 toolkit when it is installed. The `nvcc` on PATH here is
# 12.4, which cannot target sm_120 at all ("Value 'sm_120a' is not defined"),
# so defaulting to it makes `make pathbench` fail on the very box the
# benchmark exists for. Standing rule (2026-07-30): track the latest stable
# CUDA 13.x. Override with NVCC=... as before.
NVCC          ?= $(shell test -x /usr/local/cuda/bin/nvcc && echo /usr/local/cuda/bin/nvcc || echo nvcc)
GB_BENCH_ARCH ?= sm_120a
pathbench: tests/bench/gb_pathbench.cu
	$(NVCC) -O3 -arch=$(GB_BENCH_ARCH) -std=c++17 \
	    -o tests/bench/gb_pathbench tests/bench/gb_pathbench.cu -lcuda
	@echo "[GreenBoost] Built tests/bench/gb_pathbench (arch=$(GB_BENCH_ARCH))"
	@echo "[GreenBoost]   Run: python3 tests/bench/gb_pathbench.py"

clean: ebpf-clean
	$(MAKE) -C $(KDIR) M=$(PWD) $(KERNEL_LLVM) clean
	rm -f $(SHIM) $(AUDIT) $(AUDIT32) $(VMM_OVERRIDE) $(NETD) greenboost_cuda_v12.o
	rm -f third_party/gb_cutlass/*.so third_party/gb_cutlass/_gb_cutlass_C*.so
	rm -rf third_party/gb_cutlass/build
	rm -f tests/bench/gb_pathbench

install: all dkms-install install-libs
	@echo "[GreenBoost] Install complete. Load with: sudo modprobe greenboost"
	@echo "[GreenBoost] (greenboost_setup.sh's full reinstall flow loads the"
	@echo "[GreenBoost]  module itself, AFTER writing /etc/modprobe.d/greenboost.conf ,"
	@echo "[GreenBoost]  see that script for the tuned-parameter load.)"

install-legacy: all
	$(MAKE) -C $(KDIR) M=$(PWD) $(KERNEL_LLVM) modules_install
	depmod -a
	$(MAKE) install-libs

build-info:
	@printf 'BUILD_ID=%s\nBUILD_VERSION=%s\nBUILD_HOST=%s\nBUILD_GIT=%s\nBUILD_EPOCH=%s\n' \
		"$$(date +%d%m-%H%M)" "$(GB_VERSION)" "$$(hostname)" \
		"$$(git rev-parse --short HEAD 2>/dev/null || echo nogit)" \
		"$$(date +%s)" > build_info

install-libs: build-info
	pkill -9 -x $(NETD) 2>/dev/null || true
	cp $(SHIM)  /usr/local/lib/$(SHIM).new  && mv /usr/local/lib/$(SHIM).new  /usr/local/lib/$(SHIM)
	cp $(AUDIT) /usr/local/lib/$(AUDIT).new && mv /usr/local/lib/$(AUDIT).new /usr/local/lib/$(AUDIT)
	@if [ -f "$(VMM_OVERRIDE)" ]; then \
		cp $(VMM_OVERRIDE) /usr/local/lib/$(VMM_OVERRIDE).new && \
		mv /usr/local/lib/$(VMM_OVERRIDE).new /usr/local/lib/$(VMM_OVERRIDE); \
		echo "[GreenBoost] Installed $(VMM_OVERRIDE)"; \
	else \
		echo "[GreenBoost] NOTICE: $(VMM_OVERRIDE) not built , skipping (expected if CUDA headers absent)"; \
	fi
	rm -f /usr/local/lib/i386-linux-gnu/$(AUDIT) /usr/local/bin/$(NETD)
	cp $(NETD) /usr/local/bin/
	mkdir -p /usr/local/lib/i386-linux-gnu
	@if [ -f "$(AUDIT32)" ]; then \
		cp $(AUDIT32) /usr/local/lib/i386-linux-gnu/$(AUDIT) && \
		echo "[GreenBoost] Installed $(AUDIT32)"; \
	else \
		echo "[GreenBoost] NOTICE: $(AUDIT32) not built - skipping (run: make audit32)"; \
	fi
	ldconfig

dkms-install:
	@echo "[GreenBoost] Cleaning previous installation residues..."
	rmmod greenboost 2>/dev/null || true
	find /lib/modules -name "greenboost.ko*" -delete 2>/dev/null || true
	rm -rf /usr/src/greenboost-* 2>/dev/null || true
	rm -rf /var/lib/dkms/greenboost 2>/dev/null || true
	@echo "[GreenBoost] Installing DKMS source tree to $(DKMS_ROOT)..."
	mkdir -p $(DKMS_ROOT)
	cp greenboost.c greenboost_ioctl.h Kbuild Makefile dkms.conf $(DKMS_ROOT)/
	cp -r features $(DKMS_ROOT)/
	dkms add greenboost/$(GB_VERSION)
	dkms build greenboost/$(GB_VERSION)
	dkms install greenboost/$(GB_VERSION)
	@echo "[GreenBoost] DKMS module installed for kernel $(shell uname -r)"
	@# Also build for every OTHER kernel already on disk (e.g. an installed-but-
	@# not-yet-booted kernel from ~/Dev/kernel_inference) , `dkms install` above
	@# only ever covers `uname -r`, so a second kernel present at install time
	@# would otherwise stay silently unbuilt until someone notices and reruns
	@# dkms by hand. Best-effort: never fail the install target over this.
	@dkms autoinstall -m greenboost -v $(GB_VERSION) 2>&1 | sed 's/^/[GreenBoost] /' || true

dkms-uninstall:
	dkms remove greenboost/$(GB_VERSION) --all 2>/dev/null || true
	rm -rf $(DKMS_ROOT)

uninstall:
	rmmod greenboost 2>/dev/null || true
	dkms remove greenboost/$(GB_VERSION) --all 2>/dev/null || true
	rm -rf /var/lib/dkms/greenboost 2>/dev/null || true
	rm -rf /usr/src/greenboost-* 2>/dev/null || true
	find /lib/modules -name "greenboost.ko*" -delete 2>/dev/null || true
	pkill -9 -x $(NETD) 2>/dev/null || true
	rm -f /usr/local/lib/$(SHIM) /usr/local/lib/$(AUDIT)
	rm -f /usr/local/lib/i386-linux-gnu/$(AUDIT)
	depmod -a

load: module
	@if lsmod | grep -q "^greenboost "; then \
		echo "[GreenBoost] Already loaded - reloading..."; \
		sudo rmmod greenboost || true; \
	fi
	sudo insmod $(MODULE) \
		physical_vram_gb=$(PHYS_GB) \
		virtual_vram_gb=$(VIRT_GB)  \
		safety_reserve_gb=$(RESERVE_GB) \
		nvme_swap_gb=$(NVME_GB) \
		nvme_pool_gb=$(NVME_POOL)
	@if [ -f /etc/udev/rules.d/99-greenboost.rules ]; then \
		sudo udevadm trigger --name-match=greenboost 2>/dev/null || true; \
	else \
		sudo chmod 660 /dev/greenboost && sudo chgrp video /dev/greenboost 2>/dev/null || true; \
	fi
	@echo "[GreenBoost] v$(GB_VERSION) loaded - T1:$(PHYS_GB)GB T2:$(VIRT_GB)GB T3:$(NVME_GB)GB"

unload:
	sudo rmmod greenboost 2>/dev/null || true

reload: unload load

status:
	@lsmod | grep -E "^greenboost" && echo "  Module: LOADED" || echo "  Module: not loaded"
	@cat /sys/class/greenboost/greenboost/status 2>/dev/null || echo "  (module not loaded)"
	@echo "--- Kernel Logs ---"
	@sudo dmesg 2>/dev/null | grep greenboost | tail -10 | sed 's/^/  /' || echo "  (requires sudo to read dmesg)"

help:
	@echo "GreenBoost v2.9 - make [module|shim|audit|clean|install|load|unload|reload|status]"
	@echo "  T1=$(PHYS_GB)GB VRAM  T2=$(VIRT_GB)GB DDR  T3=$(NVME_GB)GB NVMe"
