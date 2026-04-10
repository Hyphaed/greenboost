# GreenBoost v2.8 — Kbuild rules (kernel-internal, not user-facing)
# Keeping these here prevents kernel Kbuild from auto-generating Makefile.

ccflags-y += -march=native -mtune=native
ccflags-y += -O2
ccflags-y += -fno-strict-aliasing
ccflags-y += -fno-strict-overflow
ccflags-y += -fno-delete-null-pointer-checks
ccflags-y += -funroll-loops
ccflags-y += -fno-common

# greenboost.ko — NVLink pool logic is included directly in greenboost.c
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
