# GreenBoost v2.9 - Kernel module + CUDA shim build system
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
GB_VERSION := 3.0

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

DKMS_ROOT := /usr/src/greenboost-$(GB_VERSION)

.PHONY: all module shim audit audit32 netd clean test \
        install install-legacy install-libs build-info dkms-install dkms-uninstall \
        uninstall load unload reload status help

all: module shim audit audit32 netd vmm_override

# PR-W: pytest regression suite covering wire-protocol layout, PSK file
# loading, LAN filter, and payload-size validation.  Pure-Python - no
# CUDA, no /dev/greenboost, no running daemon needed.  Fast: ~30 ms.
test:
	@cd tests && python3 -m pytest -v 2>&1 | tail -10
	@echo "[GreenBoost] regression suite passed"

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

netd: greenboost_netd.c features/net_fabric.h
	$(CC) $(COMMON_CFLAGS) -D_FORTIFY_SOURCE=2 -fstack-protector-strong \
	    -Wformat -Wformat-security \
	    -o $(NETD) greenboost_netd.c -ldl -lpthread $(NCCL_LDFLAGS)
	@echo "[GreenBoost] Built $(NETD)"

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) $(KERNEL_LLVM) clean
	rm -f $(SHIM) $(AUDIT) $(AUDIT32) $(VMM_OVERRIDE) $(NETD) greenboost_cuda_v12.o

install: all dkms-install install-libs
	@echo "[GreenBoost] Install complete. Load with: sudo modprobe greenboost"

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
		echo "[GreenBoost] NOTICE: $(VMM_OVERRIDE) not built — skipping (expected if CUDA headers absent)"; \
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
