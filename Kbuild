# GreenBoost v2.8 - Kbuild rules (kernel-internal, not user-facing)
# Keeping these here prevents kernel Kbuild from auto-generating Makefile.

ccflags-y += -O2
ccflags-y += -fno-strict-aliasing
ccflags-y += -fno-strict-overflow
ccflags-y += -fno-delete-null-pointer-checks
ccflags-y += -funroll-loops
ccflags-y += -fno-common
ccflags-y += -fprefetch-loop-arrays
ccflags-y += -Wall
ccflags-y += -Werror

# dma_buf_set_priority(): out-of-tree RFC hint from the kernel_inference
# project's dma-buf-priority-hint patch (~/Dev/kernel_inference/upstream-
# candidates/dma-buf-priority-hint), only present on kernels built with that
# patch (e.g. this box's "hyphaed" kernel), not on any stock/upstream kernel
# regardless of version - so it can't be gated by LINUX_VERSION_CODE like the
# rest of features/compat.h. Probe the target kernel's own dma-buf.h instead.
GB_HAS_DMABUF_PRIORITY := $(shell grep -q 'dma_buf_set_priority' $(srctree)/include/linux/dma-buf.h 2>/dev/null && echo 1 || echo 0)
ccflags-y += -DGB_HAS_DMABUF_PRIORITY=$(GB_HAS_DMABUF_PRIORITY)

# dma_buf_set_compression(): sibling out-of-tree RFC hint from the
# kernel_inference project's dma-buf-compressed-descriptor patch
# (~/Dev/kernel_inference/upstream-candidates/dma-buf-compressed-descriptor),
# same probe pattern as GB_HAS_DMABUF_PRIORITY above and the same reasoning
# for why LINUX_VERSION_CODE gating doesn't apply here either.
GB_HAS_DMABUF_COMPRESSION := $(shell grep -q 'dma_buf_set_compression' $(srctree)/include/linux/dma-buf.h 2>/dev/null && echo 1 || echo 0)
ccflags-y += -DGB_HAS_DMABUF_COMPRESSION=$(GB_HAS_DMABUF_COMPRESSION)

# greenboost.ko - NVLink pool logic is included directly in greenboost.c
# via #include "features/nvlink_pool.c" to avoid the Kbuild circular dependency
# that arises when greenboost.o is both the target and a listed source.
#
# MED-09: The proper fix is to rename greenboost.c → greenboost_main.c and use:
#   obj-m := greenboost.o
#   greenboost-y := greenboost_main.o features/nvlink_pool.o
# That removes the #include hack and makes nvlink_pool MODULE_PARM_DESCs visible
# to modinfo.  Until that rename happens, DO NOT add features/nvlink_pool.o here
# or MODULE_PARM_DESC symbols will be defined twice, causing a build error.
obj-m += greenboost.o

EXTRA_CFLAGS += -I$(src)/features
