/* greenboost_cuda_v12.c - libcudart.so.12 non-default version aliases
 *
 * Compiled WITHOUT -flto so that .symver inline asm directives survive into
 * the final .o and are not stripped by GCC GIMPLE recompilation during
 * link-time optimization of the main shim TU.
 *
 * Each trampoline calls the corresponding hook defined in
 * greenboost_cuda_shim.c.  The .symver directive renames it from __gb_*_v12
 * to <name>@libcudart.so.12 in the dynamic symbol table, satisfying PyTorch
 * cu128 versioned dlsym lookups (cudaMalloc@libcudart.so.12, etc.).
 */

#include <stddef.h>

typedef int                 cudaError_t;
typedef struct CUstream_st *CUstream;
typedef CUstream            cudaStream_t;
typedef struct { unsigned x, y, z; } gb_dim3;
typedef struct { unsigned x, y, z; } gb_uint3;

/* Forward declarations - implementations are in greenboost_cuda_shim.c */
extern cudaError_t cudaMalloc(void **devPtr, size_t size);
extern cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream);
extern cudaError_t cudaFree(void *devPtr);
extern cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream);  /* PR-CC */
extern cudaError_t cudaMemGetInfo(size_t *free_out, size_t *total_out);
extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind);
extern cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                   int kind, cudaStream_t stream);
extern cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                        int dstDevice, cudaStream_t stream);
extern cudaError_t cudaLaunchKernel(const void *func, gb_dim3 gridDim, gb_dim3 blockDim,
                                    void **args, size_t sharedMem, cudaStream_t stream);
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream);
extern cudaError_t cudaGetDeviceCount(int *count);
extern cudaError_t cudaSetDevice(int device);
extern cudaError_t cudaGetDeviceProperties(void *prop, int device);
extern cudaError_t cudaGetDeviceProperties_v2(void *prop, int device);
extern cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr,
                                                    unsigned int cudaVersion,
                                                    unsigned long long flags,
                                                    void *driverStatus);
extern void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                                   char *deviceFun, const char *deviceName,
                                   int thread_limit, gb_uint3 *tid, gb_uint3 *bid,
                                   gb_dim3 *bDim, gb_dim3 *gDim, int *wSize);

/* ------------------------------------------------------------------ */
/*  LTO keepalive                                                       */
/*                                                                      */
/*  This non-LTO TU calls each hook defined in the LTO greenboost_cuda  */
/*  _shim.c TU.  Because those calls are cross-TU and invisible to the  */
/*  LTO whole-program analyser, it would otherwise garbage-collect hook  */
/*  bodies that have no in-TU caller (cudaMemcpy, cudaLaunchKernel,     */
/*  cudaStreamSynchronize, __cudaRegisterFunction, cudaMemcpyAsync).    */
/*  Placing an address-taken table HERE (non-LTO) creates a linker-     */
/*  level reference that forces the LTO unit to emit and export each    */
/*  body.  The table itself is __attribute__((used)) so the no-LTO      */
/*  compiler keeps it; the linker then sees the cross-references.       */
/* ------------------------------------------------------------------ */
__attribute__((used))
void *const __gb_v12_keepalive[] = {
    (void *)cudaMalloc,           (void *)cudaMallocAsync,
    (void *)cudaFree,             (void *)cudaFreeAsync,  /* PR-CC */
    (void *)cudaMemGetInfo,
    (void *)cudaMemcpy,           (void *)cudaMemcpyAsync,
    (void *)cudaMemPrefetchAsync, (void *)cudaLaunchKernel,
    (void *)cudaStreamSynchronize,(void *)cudaGetDeviceCount,
    (void *)cudaSetDevice,        (void *)cudaGetDeviceProperties,
    (void *)cudaGetDeviceProperties_v2,
    (void *)cudaGetDriverEntryPointByVersion,
    (void *)__cudaRegisterFunction,
};

/* ------------------------------------------------------------------ */
/*  Trampolines                                                         */
/* ------------------------------------------------------------------ */

cudaError_t __gb_cudaMalloc_v12(void **devPtr, size_t size)
{ return cudaMalloc(devPtr, size); }
__asm__(".symver __gb_cudaMalloc_v12, cudaMalloc@libcudart.so.12");

cudaError_t __gb_cudaMallocAsync_v12(void **devPtr, size_t size, cudaStream_t stream)
{ return cudaMallocAsync(devPtr, size, stream); }
__asm__(".symver __gb_cudaMallocAsync_v12, cudaMallocAsync@libcudart.so.12");

cudaError_t __gb_cudaFree_v12(void *devPtr)
{ return cudaFree(devPtr); }
__asm__(".symver __gb_cudaFree_v12, cudaFree@libcudart.so.12");

/* PR-CC: PyTorch cu128 imports cudaFreeAsync@libcudart.so.12 - without
 * this trampoline + the libcudart.so.12 export, frees bypass our
 * bvmm_ht-aware dispatch (gb_bvmm_free_dispatch in shim) and
 * gb_t2_overflow_bytes drifts up by `sz` per cycle. */
cudaError_t __gb_cudaFreeAsync_v12(void *devPtr, cudaStream_t stream)
{ return cudaFreeAsync(devPtr, stream); }
__asm__(".symver __gb_cudaFreeAsync_v12, cudaFreeAsync@libcudart.so.12");

cudaError_t __gb_cudaMemGetInfo_v12(size_t *free_out, size_t *total_out)
{ return cudaMemGetInfo(free_out, total_out); }
__asm__(".symver __gb_cudaMemGetInfo_v12, cudaMemGetInfo@libcudart.so.12");

cudaError_t __gb_cudaMemcpy_v12(void *dst, const void *src, size_t count, int kind)
{ return cudaMemcpy(dst, src, count, kind); }
__asm__(".symver __gb_cudaMemcpy_v12, cudaMemcpy@libcudart.so.12");

cudaError_t __gb_cudaMemcpyAsync_v12(void *dst, const void *src, size_t count,
                                     int kind, cudaStream_t stream)
{ return cudaMemcpyAsync(dst, src, count, kind, stream); }
__asm__(".symver __gb_cudaMemcpyAsync_v12, cudaMemcpyAsync@libcudart.so.12");

cudaError_t __gb_cudaMemPrefetchAsync_v12(const void *devPtr, size_t count,
                                          int dstDevice, cudaStream_t stream)
{ return cudaMemPrefetchAsync(devPtr, count, dstDevice, stream); }
__asm__(".symver __gb_cudaMemPrefetchAsync_v12, cudaMemPrefetchAsync@libcudart.so.12");

cudaError_t __gb_cudaLaunchKernel_v12(const void *func, gb_dim3 gridDim, gb_dim3 blockDim,
                                      void **args, size_t sharedMem, cudaStream_t stream)
{ return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream); }
__asm__(".symver __gb_cudaLaunchKernel_v12, cudaLaunchKernel@libcudart.so.12");

cudaError_t __gb_cudaStreamSynchronize_v12(cudaStream_t stream)
{ return cudaStreamSynchronize(stream); }
__asm__(".symver __gb_cudaStreamSynchronize_v12, cudaStreamSynchronize@libcudart.so.12");

cudaError_t __gb_cudaGetDeviceCount_v12(int *count)
{ return cudaGetDeviceCount(count); }
__asm__(".symver __gb_cudaGetDeviceCount_v12, cudaGetDeviceCount@libcudart.so.12");

cudaError_t __gb_cudaSetDevice_v12(int device)
{ return cudaSetDevice(device); }
__asm__(".symver __gb_cudaSetDevice_v12, cudaSetDevice@libcudart.so.12");

cudaError_t __gb_cudaGetDeviceProperties_v12(void *prop, int device)
{ return cudaGetDeviceProperties(prop, device); }
__asm__(".symver __gb_cudaGetDeviceProperties_v12, cudaGetDeviceProperties@libcudart.so.12");

cudaError_t __gb_cudaGetDeviceProperties_v2_v12(void *prop, int device)
{ return cudaGetDeviceProperties_v2(prop, device); }
__asm__(".symver __gb_cudaGetDeviceProperties_v2_v12, cudaGetDeviceProperties_v2@libcudart.so.12");

cudaError_t __gb_cudaGetDriverEntryPointByVersion_v12(const char *symbol, void **funcPtr,
                                                      unsigned int cudaVersion,
                                                      unsigned long long flags,
                                                      void *driverStatus)
{ return cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion, flags, driverStatus); }
__asm__(".symver __gb_cudaGetDriverEntryPointByVersion_v12, cudaGetDriverEntryPointByVersion@libcudart.so.12");

void __gb___cudaRegisterFunction_v12(void **fatCubinHandle, const char *hostFun,
                                     char *deviceFun, const char *deviceName, int thread_limit,
                                     gb_uint3 *tid, gb_uint3 *bid,
                                     gb_dim3 *bDim, gb_dim3 *gDim, int *wSize)
{ __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit,
                          tid, bid, bDim, gDim, wSize); }
__asm__(".symver __gb___cudaRegisterFunction_v12, __cudaRegisterFunction@libcudart.so.12");
