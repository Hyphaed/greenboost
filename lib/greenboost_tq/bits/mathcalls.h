/*
 * lib/greenboost_tq/bits/mathcalls.h
 *
 * Intercept shim for <bits/mathcalls.h>.
 *
 * Problem: CUDA 13.1 (math_functions.h:629/653) declares rsqrt(double) and
 * rsqrtf(float) WITHOUT __THROW / noexcept.  glibc 2.41+ with _GNU_SOURCE
 * (IEC_60559_FUNCS_EXT_C23 block at mathcalls.h:192) then re-declares them
 * WITH noexcept(true), producing an "exception specification is incompatible"
 * error in cudafe++.
 *
 * Fix: redirect glibc's rsqrt/rsqrtf declarations to private stub names so
 * they do not collide with CUDA's declarations.  CUDA's device-builtin
 * rsqrt/rsqrtf remain the authoritative declarations.  kernels.cu does not
 * call rsqrt/rsqrtf on the host side, so the redirect has no functional
 * impact.
 *
 * This file sits in lib/greenboost_tq/ which is listed first in the -I path;
 * gcc therefore finds it before /usr/include/x86_64-linux-gnu/bits/mathcalls.h.
 * It has no include guard intentionally — glibc includes mathcalls.h multiple
 * times (double / float / long-double passes) and so must we.
 */
#define rsqrt    __gb_glibc_rsqrt_redir_
#define rsqrtf   __gb_glibc_rsqrtf_redir_
#define __rsqrt  __gb_glibc___rsqrt_redir_
#define __rsqrtf __gb_glibc___rsqrtf_redir_
#include_next <bits/mathcalls.h>
#undef rsqrt
#undef rsqrtf
#undef __rsqrt
#undef __rsqrtf
