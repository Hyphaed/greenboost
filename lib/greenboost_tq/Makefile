# GreenBoost TurboQuant library — CUDA build
# Builds libgreenboost_tq.so from kernels.cu

TQ_LIB := libgreenboost_tq.so
TQ_SRC := kernels.cu

NVCC    ?= nvcc
NVCCFLAGS := -shared -Xcompiler -fPIC -O3 --gpu-architecture=native
NVCCFLAGS += -Xcompiler -fvisibility=hidden
NVCCFLAGS += -I.
# Link pthreads for the host-side init mutex
NVCCFLAGS += -Xlinker -lpthread

.PHONY: all clean

all: $(TQ_LIB)

$(TQ_LIB): $(TQ_SRC) greenboost_tq.h turbo_types.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	@echo "[GreenBoost] Built $(TQ_LIB)"

clean:
	rm -f $(TQ_LIB)
