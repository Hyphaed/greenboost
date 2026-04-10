# GreenBoost v2.8 — Kernel module + CUDA shim build system
# Author: Ferran Duarri
#
# Kbuild file handles kernel-internal rules (obj-m, ccflags-y).
# This Makefile handles user-facing targets (module, install, load, etc.)
# Keeping them separate prevents the kernel Kbuild from overwriting this file.

KDIR    := /lib/modules/$(shell uname -r)/build
PWD     := $(shell pwd)
CC      := gcc
SHIM    := libgreenboost_cuda.so
AUDIT   := libgreenboost_audit.so
AUDIT32 := libgreenboost_audit32.so
VULKAN  := libVkLayer_greenboost.so
MODULE  := greenboost.ko
GB_VERSION := 2.8

PHYS_GB    ?= 0
VIRT_GB    ?= 0
RESERVE_GB ?= 8
NVME_GB    ?= 0
# MIN-04: Module default for nvme_pool_gb is 0 (disabled).
# Override with: make load NVME_POOL=N   to enable T3 on systems with NVMe swap.
NVME_POOL  ?= 0

# Vulkan SDK includes — auto-detected from $VULKAN_SDK env var, pkg-config, or
# standard system path. Override: make VULKAN_SDK_INCLUDES=/path/to/vulkan/include
VULKAN_SDK_INCLUDES := $(if $(VULKAN_SDK),$(VULKAN_SDK)/include,\
  $(shell pkg-config --variable=includedir vulkan 2>/dev/null || echo /usr/include))

HAS_AVX2 := $(shell grep -qw avx2 /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
COMMON_CFLAGS := -march=native -mtune=native -O3 -funroll-loops -std=gnu11
COMMON_CFLAGS += -flto -fvisibility=hidden -ffunction-sections -fdata-sections
ifeq ($(HAS_AVX2),1)
COMMON_CFLAGS += -mavx2
endif

# 32-bit header probe — needed for audit32 on modern Ubuntu (libc6-dev-i386 / gcc-multilib)
HAS_32BIT_HEADERS := $(shell test -f /usr/include/i386-linux-gnu/bits/wordsize.h && echo 1 || echo 0)
# AUDIT32_CFLAGS: i686 build — no -march=native/-mtune=native (incompatible with -m32),
# no -mavx2 (64-bit era), adds i386 multilib include path.
AUDIT32_CFLAGS := -march=i686 -O3 -funroll-loops -std=gnu11
AUDIT32_CFLAGS += -flto -fvisibility=hidden -ffunction-sections -fdata-sections
AUDIT32_CFLAGS += -I/usr/include/i386-linux-gnu
# Shim uses the version script (local: *) for symbol hiding instead of
# -fvisibility=hidden, so that version-script global: declarations actually
# export the hook functions into the dynamic symbol table.
SHIM_CFLAGS   := -march=native -mtune=native -O3 -funroll-loops -std=gnu11
SHIM_CFLAGS   += -flto -ffunction-sections -fdata-sections
ifeq ($(HAS_AVX2),1)
SHIM_CFLAGS   += -mavx2
endif
SHIM_LDFLAGS  := -Wl,--gc-sections -Wl,--as-needed

DKMS_ROOT := /usr/src/greenboost-$(GB_VERSION)

TQ_LIB := lib/greenboost_tq/libgreenboost_tq.so
TQ_SRC  := lib/greenboost_tq/kernels.cu

# Auto-detect nvcc from standard CUDA toolkit paths if not on PATH.
NVCC ?= $(firstword $(wildcard \
    $(addsuffix /bin/nvcc, \
        $(CUDA_HOME) $(CUDA_PATH) \
        /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-12 \
        /opt/cuda /usr/cuda)) \
    nvcc)

NVCC_OK := $(shell $(NVCC) --version >/dev/null 2>&1 && echo 1 || echo 0)

# Probe for the best GPU architecture: prefer native, fall back to the highest
# arch the installed CUDA toolkit supports. Handles the case where the GPU is
# newer than the toolkit (e.g. Blackwell/RTX 50xx with CUDA 12.x).
ifeq ($(NVCC_OK),1)
  _TQ_TMPDIR  := $(shell mktemp -d /tmp/gb_nvcc_probe.XXXXXX)
  _NATIVE_OK  := $(shell $(NVCC) -x cu /dev/null -o $(_TQ_TMPDIR)/probe --gpu-architecture=native -c 2>/dev/null && echo 1 || echo 0)
  $(shell rm -rf $(_TQ_TMPDIR))
  ifeq ($(_NATIVE_OK),1)
    GPU_ARCH := native
  else
    GPU_ARCH := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | tail -1)
  endif
endif

.PHONY: all module shim audit audit32 vulkan tq clean \
        install install-legacy install-libs dkms-install dkms-uninstall \
        uninstall load unload reload status help

all: module shim audit audit32 vulkan tq

module:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

shim: greenboost_cuda_shim.c greenboost_cuda.map
	$(CC) -shared -fPIC $(SHIM_CFLAGS) -o $(SHIM) greenboost_cuda_shim.c -ldl -lpthread \
		-Wl,--version-script=greenboost_cuda.map $(SHIM_LDFLAGS)
	@echo "[GreenBoost] Built $(SHIM)"

# SHIM_INSTALL_PATH: where libgreenboost_cuda.so will be installed.
# Override at build time: make audit SHIM_INSTALL_PATH=/usr/lib/libgreenboost_cuda.so
SHIM_INSTALL_PATH ?= /usr/local/lib/libgreenboost_cuda.so

audit: greenboost_audit.c
	$(CC) -shared -fPIC $(COMMON_CFLAGS) -DSHIM_PATH=\"$(SHIM_INSTALL_PATH)\" \
		-o $(AUDIT) greenboost_audit.c -ldl
	@echo "[GreenBoost] Built $(AUDIT) (SHIM_PATH=$(SHIM_INSTALL_PATH))"

audit32: greenboost_audit.c
ifeq ($(HAS_32BIT_HEADERS),1)
	$(CC) -m32 -shared -fPIC $(AUDIT32_CFLAGS) -DSHIM_PATH=\"$(SHIM_INSTALL_PATH)\" \
		-o $(AUDIT32) greenboost_audit.c -ldl
	@echo "[GreenBoost] Built $(AUDIT32) (i386, SHIM_PATH=$(SHIM_INSTALL_PATH))"
else
	@echo "[GreenBoost] NOTICE: 32-bit headers not found — skipping $(AUDIT32)"
	@echo "[GreenBoost]   Install with: sudo apt install gcc-multilib libc6-dev-i386"
	@echo "[GreenBoost]   Then re-run: make audit32"
endif

vulkan: greenboost_vulkan_layer.c greenboost_ioctl.h
	$(CC) -shared -fPIC $(COMMON_CFLAGS) -o $(VULKAN) greenboost_vulkan_layer.c \
		-I$(VULKAN_SDK_INCLUDES) -lpthread
	@echo "[GreenBoost] Built $(VULKAN)"

tq:
ifeq ($(NVCC_OK),1)
ifneq ($(GPU_ARCH),)
	$(MAKE) $(TQ_LIB)
else
	@echo "[GreenBoost] NOTICE: no supported GPU architecture found — skipping libgreenboost_tq.so"
	@echo "[GreenBoost]   Update CUDA toolkit to support your GPU, then re-run: make tq"
endif
else
	@echo "[GreenBoost] NOTICE: nvcc not found — skipping libgreenboost_tq.so (TurboQuant disabled)"
	@echo "[GreenBoost]   Install CUDA toolkit and re-run: make tq"
endif

$(TQ_LIB): $(TQ_SRC) lib/greenboost_tq/greenboost_tq.h lib/greenboost_tq/turbo_types.h
	$(NVCC) -shared -o $@ $< -Xcompiler -fPIC -O3 --gpu-architecture=$(GPU_ARCH) \
		-Ilib/greenboost_tq -Xlinker -lpthread
	@echo "[GreenBoost] Built $(TQ_LIB) (arch=$(GPU_ARCH))"

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f $(SHIM) $(AUDIT) $(AUDIT32) $(VULKAN) $(TQ_LIB)

install: all dkms-install install-libs
	@echo "[GreenBoost] Install complete. Load with: sudo modprobe greenboost"

install-legacy: all
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a
	$(MAKE) install-libs

install-libs:
	rm -f /usr/local/lib/$(SHIM) /usr/local/lib/$(AUDIT) /usr/local/lib/$(VULKAN)
	rm -f /usr/local/lib/i386-linux-gnu/$(AUDIT)
	cp $(SHIM) /usr/local/lib/
	cp $(AUDIT) /usr/local/lib/
	cp $(VULKAN) /usr/local/lib/
	mkdir -p /usr/local/lib/i386-linux-gnu
	@if [ -f "$(AUDIT32)" ]; then \
		cp $(AUDIT32) /usr/local/lib/i386-linux-gnu/$(AUDIT) && \
		echo "[GreenBoost] Installed $(AUDIT32)"; \
	else \
		echo "[GreenBoost] NOTICE: $(AUDIT32) not built — skipping (run: make audit32)"; \
	fi
	@if [ -f "$(TQ_LIB)" ]; then \
		cp $(TQ_LIB) /usr/local/lib/ && echo "[GreenBoost] Installed libgreenboost_tq.so"; \
	else \
		echo "[GreenBoost] NOTICE: $(TQ_LIB) not built — skipping (run: make tq)"; \
	fi
	ldconfig

dkms-install:
	@echo "[GreenBoost] Installing DKMS source tree to $(DKMS_ROOT)..."
	mkdir -p $(DKMS_ROOT)
	cp greenboost.c greenboost_ioctl.h Kbuild Makefile dkms.conf $(DKMS_ROOT)/
	cp -r features $(DKMS_ROOT)/
	@if [ -f "$(TQ_LIB)" ]; then \
		cp $(TQ_LIB) /usr/local/lib/ && echo "[GreenBoost] Installed libgreenboost_tq.so (dkms-install)"; \
	fi
	dkms remove greenboost/$(GB_VERSION) --all 2>/dev/null || true
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
	rm -rf $(DKMS_ROOT) 2>/dev/null || true
	rm -f /lib/modules/$(shell uname -r)/extra/greenboost.ko
	rm -f /usr/local/lib/$(SHIM) /usr/local/lib/$(AUDIT) /usr/local/lib/$(VULKAN)
	rm -f /usr/local/lib/i386-linux-gnu/$(AUDIT)
	rm -f /etc/vulkan/implicit_layer.d/VkLayer_greenboost.json
	depmod -a

load: module
	@if lsmod | grep -q "^greenboost "; then \
		echo "[GreenBoost] Already loaded — reloading..."; \
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
	@echo "[GreenBoost] v$(GB_VERSION) loaded — T1:$(PHYS_GB)GB T2:$(VIRT_GB)GB T3:$(NVME_GB)GB"

unload:
	sudo rmmod greenboost 2>/dev/null || true

reload: unload load

status:
	@lsmod | grep -E "^greenboost" && echo "  Module: LOADED" || echo "  Module: not loaded"
	@cat /sys/class/greenboost/greenboost/status 2>/dev/null || echo "  (module not loaded)"
	@echo "--- Kernel Logs ---"
	@sudo dmesg 2>/dev/null | grep greenboost | tail -10 | sed 's/^/  /' || echo "  (requires sudo to read dmesg)"

help:
	@echo "GreenBoost v2.8 — make [module|shim|audit|vulkan|clean|install|load|unload|reload|status]"
	@echo "  T1=$(PHYS_GB)GB VRAM  T2=$(VIRT_GB)GB DDR  T3=$(NVME_GB)GB NVMe"
