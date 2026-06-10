# GreenBoost - Error-Code Conventions

Audit reference: AUDIT_2026-05-13.md F-L1-35 / F-L3-16 / F-L5-32 / §9.

Each layer uses ONE canonical error type. Translation happens at layer
boundaries, never mid-function.

## The Rule

| Layer | Canonical type | Examples |
|---|---|---|
| Kernel module (`greenboost.c`) | `-errno` (negative `int`) | `-ENOMEM`, `-EINVAL`, `-EPERM`, `-ENODEV` |
| Wire protocol + daemon (`greenboost_netd.c`, `features/net_fabric.h`) | `enum gb_net_status` | `GB_STATUS_OK`, `GB_STATUS_ERR_OOM`, `GB_STATUS_ERR_INVALID`, `GB_STATUS_ERR_CUDA`, `GB_STATUS_ERR_PROTO`, `GB_STATUS_ERR_REJECTED`, `GB_STATUS_ERR_NCCL`, `GB_STATUS_ERR_THROTTLE` |
| CUDA shim hooks (`greenboost_cuda_shim.c`) | `CUresult` for `cu*` hooks; `cudaError_t` for `cuda*` hooks | `CUDA_SUCCESS`, `CUDA_ERROR_OUT_OF_MEMORY`, `CUDA_ERROR_INVALID_VALUE`, `cudaSuccess`, `cudaErrorMemoryAllocation` |
| Host-client (`greenboost_netc.c`) | `int` - 0 on success, `-1` on failure (plus optional `gb_net_status` payload from the wire) | `0`, `-1` |
| Python tooling (`greenboost_exporter.py`, `gb_feeder_diag.py`, `greenboost_builder.py`) | Native exceptions - never silently coerce | `OSError`, `socket.timeout`, `struct.error`, `subprocess.CalledProcessError` |

## Translation Boundaries

Convert exactly **once**, at the moment a value crosses a layer:

```c
/* Wire ↔ CUDA hook boundary (greenboost_cuda_shim.c) */
static CUresult gb_status_to_cuda(uint32_t s) {
    switch (s) {
    case GB_STATUS_OK:           return CUDA_SUCCESS;
    case GB_STATUS_ERR_OOM:      return CUDA_ERROR_OUT_OF_MEMORY;
    case GB_STATUS_ERR_INVALID:  return CUDA_ERROR_INVALID_VALUE;
    case GB_STATUS_ERR_CUDA:     return CUDA_ERROR_UNKNOWN;
    case GB_STATUS_ERR_REJECTED: return CUDA_ERROR_NOT_PERMITTED;
    case GB_STATUS_ERR_NCCL:     return CUDA_ERROR_NOT_SUPPORTED;
    case GB_STATUS_ERR_THROTTLE: return CUDA_ERROR_NOT_READY;
    case GB_STATUS_ERR_PROTO:    return CUDA_ERROR_NOT_SUPPORTED;
    default:                     return CUDA_ERROR_UNKNOWN;
    }
}

/* Kernel ↔ ioctl boundary (greenboost.c)
 * - the kernel module already returns -errno via the file_operations ABI,
 *   userspace gets the value through ioctl(2)'s return code, no extra
 *   wrapping needed. */
```

## What "mixed types in one function" looks like

Bad - three different conventions in one chain:

```c
/* DO NOT DO THIS */
static int handle_alloc(...) {
    void *p;
    cudaError_t err = cudaMalloc(&p, sz);  /* cudaError_t  */
    if (err != cudaSuccess) {
        return -ENOMEM;                    /* -errno       */
    }
    /* …later… */
    if (overflow) {
        resp.status = GB_STATUS_ERR_OOM;   /* gb_net_status */
    }
    /* …caller does `if (err < 0)` and gets it wrong on CUDA_ERROR_*… */
}
```

Good - typed locally, translated only at the response boundary:

```c
static int handle_alloc(struct client *cli, ...) {
    void *p;
    cudaError_t cuerr = cudaMalloc(&p, sz);
    uint32_t status   = (cuerr == cudaSuccess) ? GB_STATUS_OK
                                               : GB_STATUS_ERR_OOM;
    /* …later, single response struct uses gb_net_status… */
    resp.status = status;
    return send_msg(cli->fd, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                    &resp, sizeof(resp));
}
```

## How to check compliance

```bash
# Find sites that mix -ENOMEM and GB_STATUS_ERR_OOM in one file:
grep -lE '(-ENOMEM|GB_STATUS_ERR_OOM)' greenboost_*.c features/*.h \
    | xargs grep -lE 'CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation' 2>/dev/null

# Should match at most the dedicated translation helper(s) - anywhere else
# is a sign that a function is straddling layers.
```

## Status of the migration

As of `AUDIT_2026-05-13.md` (commit landing this file):

- ✅ Wire protocol uses `gb_net_status` consistently (clean).
- ✅ Kernel module uses `-errno` consistently (clean).
- ⚠ Daemon (`greenboost_netd.c`) mixes `-ENOMEM` (internal alloc failures)
  with `GB_STATUS_ERR_OOM` (response payloads).  Internal `-ENOMEM` returns
  are local to one function and never reach the wire - current behaviour
  is correct, but a single-function audit + comment pass would make this
  obvious to new readers.
- ⚠ CUDA shim has a handful of internal helpers returning `int` for
  "did the overflow allocation succeed".  Those should consistently
  return `CUresult` so the immediate caller can pass through to a hook
  return without translation.

These remaining inconsistencies are tracked as F-L1-35 / F-L3-16 and are
non-breaking - fix opportunistically when touching the relevant function.
