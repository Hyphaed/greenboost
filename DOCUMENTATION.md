# GreenBoost v2.10 - integration guide for inference tools

Check [GREENBOOST_COMMANDS.md](GREENBOOST_COMMANDS.md) for all the available commands
Check [ARCHITECTURE.md](ARCHITECTURE.md) to know about greenboost architecture
Check [CHANGELOG.md](CHANGELOG.md) for what changed in each version

---

## Reading this document

Two audiences read this file:

- **Newcomers** who installed GreenBoost and want to make a specific
  inference tool work. Jump straight to your tool's section
  ([Ollama](#ollama), [vLLM](#vllm), [PyTorch](#pytorch-scripts),
  [Hugging Face](#hugging-face-transformers), …). The first paragraph
  of each section is a copy-pasteable recipe; everything after that is
  optional tuning.
- **Engineers integrating GreenBoost into infrastructure** who want
  the architectural model - phase detector, KV-cache pinning, cluster
  fabric, security. Read [§ How GreenBoost hooks in](#how-greenboost-hooks-in)
  first, then the [§ Cluster](#cluster--connect-remote-machines-as-gpu-memory-and-compute-feeders)
  and [§ Debug Vitals](#debug-vitals) sections.

If you only want to *use* GreenBoost with one inference tool, you can
skip the architecture sections entirely. The shim is transparent - your
existing Python script does not change.

---

## What's new in v2.10

The 2.10 cycle was a multi-month hardening campaign on the cluster
fabric and the LD_PRELOAD shim. No new user-visible APIs to learn, but
plenty of fixes you'll notice in production:

**Security**
- Cluster fabric upgraded from proto v3 to **proto v4** with mutual
  authentication: both the host *and* each feeder prove knowledge of
  the PSK. A man-in-the-middle who only replays a prior handshake can
  no longer impersonate either side.
- Every message after the handshake carries a truncated HMAC-SHA256
  computed over (header ‖ payload) keyed by an HKDF-derived session
  key. Tampering is detected at receive time, not after parsing.
- A v4 host still talks to a v3 feeder if `GREENBOOST_PSK_V4=0` is set
  (so you can roll feeders one at a time); default is v4-only.

**Robustness**
- 117-test pytest suite covering wire-protocol, PSK loading, LAN
  filter, HKDF, exported symbols, and end-to-end handshake.
- Kernel-module DMA path holds a `dma_resv` reservation across every
  device mapping, with `active_mappings` reference counting so the
  eviction worker can never free memory the GPU is reading.
- `cuFuncGetParamInfo` (CUDA 12.3+) probes kernels for argument
  arity, so `cuLaunchKernel` argument scanning has correct bounds
  instead of guessing.
- The hot allocation path is fully big-endian-safe - every wire field
  is read via `GB_LE_U16/U32/U64` accessors. The full BE port still
  requires opt-in (`-DGREENBOOST_BE_INCOMPLETE=1`).

**New tools**
- `greenboost stability` - a Python long-running monitor that watches
  `/run/greenboost/shim_stats` for monotone-counter regressions,
  tier-gauge leaks (memory still allocated after idle), and
  fragmentation drift over hours of inference.
- `greenboost feeders diag` - host-side T1/T2/T3 + compute test
  against a remote feeder.
- A worker-pool scaffold inside `greenboost-netd` (opt in with
  `GREENBOOST_WORKERS=N`) so a slow CUDA op on one client no longer
  blocks the whole reactor.

**Decoupled**
- The experimental Vulkan layer that let games use the GreenBoost
  pool has been moved into a sister project: **GreenBoost Gaming
  Suite**. The CUDA-inference path stays small and auditable; gamers
  get a GTK4 GUI instead of CLI. See [§ Sister project: GreenBoost
  Gaming Suite](#sister-project-greenboost-gaming-suite).

---

## How GreenBoost hooks in

GreenBoost works by intercepting `cudaMalloc` and related CUDA symbols via `LD_PRELOAD`, plus intercepting `dlsym` so that the virtual VRAM total (T1 VRAM + T2 DDR pool, computed from detected hardware) is returned to any app that queries VRAM size at runtime through `dlopen`+`dlsym`.

The shim loads system-wide via `/etc/ld.so.preload` but stays **inert** until
`GREENBOOST_ACTIVE=1` is set. In **interactive login shells** (terminal, SSH),
`/etc/profile.d/greenboost.sh` exports `GREENBOOST_ACTIVE=1` automatically - no wrapper
needed. Just run your script directly:

```bash
python your_script.py               # shim already active in login shells
python -m vllm.entrypoints.openai.api_server ...
```

In **non-login contexts** (cron jobs, Docker entrypoints, sudo scripts, spawned subshells)
profile.d is not sourced. Use the wrapper or set the variable explicitly:

```bash
# Wrapper - sets GREENBOOST_ACTIVE=1 for one command
greenboost run python your_script.py

# Or inline
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so python your_script.py

# Or in a systemd unit / Docker environment
Environment="GREENBOOST_ACTIVE=1"
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
```

---

## Ollama

Handled automatically by `install-sys-configs`, which injects the shim and
`GREENBOOST_ACTIVE=1` into the systemd unit. No manual wrapper needed.

```bash
ollama run glm-4.7-flash:q8_0   # GreenBoost is transparent
```

If running Ollama outside systemd:

```bash
greenboost run ollama serve
```

---

## vLLM

vLLM loads libcuda lazily through PyTorch (`torch.cuda` → `ctypes.CDLL`).

```bash
greenboost run python -m vllm.entrypoints.openai.api_server \
    --model /opt/models/glm-4.7-flash-hf \
    --dtype float16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95
```

As a systemd service, add to the unit:

```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
Environment="GREENBOOST_ACTIVE=1"
```

---

## PyTorch scripts

Any `import torch; torch.cuda.*` call triggers lazy loading of libcuda via ctypes.

```bash
greenboost run python your_inference_script.py
```

Or inline without the wrapper:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python your_script.py
```

---

## text-generation-inference (TGI)

TGI uses a PyTorch backend - same lazy-CUDA pattern:

```bash
greenboost run text-generation-launcher \
    --model-id /opt/models/glm-4.7-flash-hf \
    --num-shard 1 \
    --max-total-tokens 131072
```

As a systemd service:

```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
Environment="GREENBOOST_ACTIVE=1"
```

---

## CTranslate2

CTranslate2 loads libcudart via ctypes in its Python bindings:

```bash
greenboost run python your_ctranslate2_script.py
```

---

## Hugging Face Transformers

Transformers loads CUDA through PyTorch - same lazy-CUDA pattern as vLLM and TGI.

```bash
greenboost run python your_transformers_script.py
```

With `device_map="auto"`, Transformers queries available VRAM before placing layers.
GreenBoost reports the detected T1+T2 total, so the full model is placed on the "GPU"
(T1+T2) instead of being split to CPU:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/glm-4.7-flash-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",          # sees T1+T2 total - loads entire model onto T1+T2
)
tokenizer = AutoTokenizer.from_pretrained("/opt/models/glm-4.7-flash-hf")

messages = [{"role": "user", "content": "Hello!"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
).to(model.device)

output = model.generate(input_ids, max_new_tokens=300)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Pipeline API works the same way:

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="/opt/models/glm-4.7-flash-hf",
    torch_dtype="bfloat16",
    device_map="auto",
)
print(pipe("Hello!", max_new_tokens=200)[0]["generated_text"])
```

Downloading a model then running it in one shot:

```bash
greenboost run python - <<'EOF'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

snapshot_download("THUDM/glm-4.7-flash-hf", local_dir="/opt/models/glm-4.7-flash-hf")

model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/glm-4.7-flash-hf", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/opt/models/glm-4.7-flash-hf")
ids = tokenizer("Hello!", return_tensors="pt").input_ids.to(model.device)
print(tokenizer.decode(model.generate(ids, max_new_tokens=100)[0]))
EOF
```

When using the GreenBoost venv:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    /opt/greenboost/venv/bin/python your_transformers_script.py
```

---

## ExLlamaV3

ExLlamaV3 loads CUDA through PyTorch:

```bash
greenboost run python your_exllama_script.py
```

When using the GreenBoost venv:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    /opt/greenboost/venv/bin/python your_exllama_script.py
```

The GreenBoost-patched KV cache layer (`CacheLayer_greenboost`) allocates directly
from `/dev/greenboost` and does not go through the shim's cudaMalloc path for cache
tensors - `GREENBOOST_ACTIVE=1` is still needed for virtual VRAM reporting.

---

## PyTorch - direct tensor allocation control

PyTorch loads libcuda lazily via `ctypes.CDLL` when `torch.cuda` is first accessed.
GreenBoost intercepts from that point forward.

```bash
greenboost run python your_script.py
# or inline:
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so python your_script.py
```

### Explicit KV cache flagging from Python

For custom inference loops where you know exactly which tensors are KV cache, use
`GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY` via the MADVISE IOCTL so the kernel
freezes them in T2's LRU and refuses T3 spill:

```python
import fcntl, struct, os
from pathlib import Path

# GB_IOCTL_MADVISE = _IOW('G', 4, struct gb_madvise_req)  - 8 bytes
_IOC_MADVISE     = (1 << 30) | (ord('G') << 8) | 4 | (8 << 16)
GB_MADVISE_T1_PREFER = 3   # mark as T1-priority: freeze in T2 LRU, refuse T3 spill

def gb_mark_kv(buf_fd: int) -> None:
    """
    Tell GreenBoost that buf_fd (DMA-BUF fd from GB_IOCTL_ALLOC) is a KV cache buffer.
    Kernel will: freeze in T2 LRU, refuse T3 spill, set t1_priority.
    buf_fd is the fd returned by GB_IOCTL_ALLOC (stored in gb_alloc_req.fd).
    """
    dev = Path("/dev/greenboost")
    if not dev.exists():
        return
    # struct gb_madvise_req { s32 buf_id; u32 advise; }
    buf = struct.pack("iI", buf_fd, GB_MADVISE_T1_PREFER)
    fd  = os.open(str(dev), os.O_RDWR)
    try:
        fcntl.ioctl(fd, _IOC_MADVISE, buf)
    finally:
        os.close(fd)

# Usage with PyTorch (via ExLlamaV3 native IOCTL path)
# buf_fd is obtained from GB_IOCTL_ALLOC, not from cudaMalloc.
# For pure-PyTorch workflows, use GREENBOOST_KV_OVERFLOW=1 or the
# phase detector (automatic) instead of calling this directly.
```

### Optimal loading order

GreenBoost's phase detector expects: **weights first, KV cache second**.
Respect this order to get maximum T1 utilization:

```python
import torch

# 1. Load model weights (phase = MODEL_LOAD - reserve inactive, weights fill T1)
model = MyModel().cuda()
model.load_state_dict(torch.load("model.pt", map_location="cuda"))

# 2. Allocate KV cache (phase = INFERENCE - reserve activates, KV lands in T1)
kv_cache = torch.zeros(batch, n_layers, seq_len, d_model, device="cuda")
# ↑ GreenBoost auto-classifies this as KV via quiet-gap heuristic
#   OR call gb_mark_kv() above for certainty

# 3. Run inference
output = model.generate(input_ids, past_key_values=kv_cache)
```

If your script loads KV cache before or interleaved with weights, set
`GREENBOOST_KV_OVERFLOW=1` to disable the heuristic.

---

## Hugging Face Transformers - direct KV cache control

Transformers uses PyTorch's CUDA allocator; GreenBoost intercepts all `cudaMalloc`
calls. `device_map="auto"` queries VRAM size - GreenBoost reports the detected T1+T2
total, so the full model is placed on T1+T2 instead of being split to CPU.

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python inference.py
```

### Pinning the KV cache to T1

Transformers' `DynamicCache` allocates keys/values lazily during generation.
To ensure they land in T1 rather than T2, set a generous KV reserve **before**
generation starts and tell GreenBoost the phase is INFERENCE:

```python
import os
os.environ["GREENBOOST_KV_RESERVE_MB"] = "4096"   # 4 GB reserved for KV
os.environ["GREENBOOST_KV_OVERFLOW"]   = "0"       # use phase detector (default)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/glm-4.7-flash-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",          # sees T1+T2 total - all layers on GPU
    attn_implementation="flash_attention_2",   # requires OLLAMA_FLASH_ATTENTION=1 equivalent
)
tokenizer = AutoTokenizer.from_pretrained("/opt/models/glm-4.7-flash-hf")

# All tokens below will have KV cache allocated in T1 (phase detector: INFERENCE)
out = model.generate(
    tokenizer("Hello!", return_tensors="pt").input_ids.cuda(),
    max_new_tokens=512,
    use_cache=True,   # ensure KV cache is used
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Mark Transformers' KV cache explicitly (advanced)

If the phase detector misclassifies activations as KV, disable it and mark manually:

```python
import os
os.environ["GREENBOOST_PHASE_DETECT"] = "0"    # disable auto-classification

from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(..., device_map="auto")

# Monkey-patch _update_causal_mask to mark KV after each alloc
_orig_forward = model.forward

def _patched_forward(*args, **kwargs):
    out = _orig_forward(*args, **kwargs)
    # Walk past_key_values and mark each tensor as KV
    if out.past_key_values:
        for layer_kv in out.past_key_values:
            for tensor in layer_kv:
                if tensor.is_cuda:
                    gb_mark_kv(tensor.data_ptr(), tensor.nbytes)
    return out

model.forward = _patched_forward
```

### Context-window KV reserve sizing

| OLLAMA_NUM_CTX / max_new_tokens | Recommended GREENBOOST_KV_RESERVE_MB |
|----------------------------------|---------------------------------------|
| ≤ 8 K tokens                    | 1024 MB                               |
| 8 K – 32 K tokens               | 2048 MB (default)                     |
| 32 K – 64 K tokens              | 4096 MB                               |
| 64 K – 128 K tokens             | 6144 MB                               |
| > 128 K tokens                  | 8192 MB                               |

---

## vLLM - direct KV cache placement

vLLM manages its own block-based KV cache allocator (PagedAttention). It pre-allocates
the entire KV cache up front from GPU memory before inference begins.

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python -m vllm.entrypoints.openai.api_server \
        --model /opt/models/glm-4.7-flash-hf \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.95 \
        --enforce-eager
```

### Why `--gpu-memory-utilization 0.95` with GreenBoost

vLLM queries `torch.cuda.mem_get_info()` to calculate how many KV blocks it can
allocate. GreenBoost intercepts this and returns T1 free space. At 0.95 utilization,
vLLM will try to allocate ~11.4 GB of KV cache from "GPU" memory - GreenBoost
routes any T1 overflow to T2 via DMA-BUF.

**Increase KV reserve before launching vLLM** so that vLLM's pre-allocated KV cache
gets priority over leftover weight fragments in T1:

```bash
# Set 8 GB reserve before starting vLLM (131K context needs ~7–9 GB KV)
GREENBOOST_KV_RESERVE_MB=8192 \
GREENBOOST_KV_OVERFLOW=1 \
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python -m vllm.entrypoints.openai.api_server \
        --model /opt/models/glm-4.7-flash-hf \
        --max-model-len 131072
```

`GREENBOOST_KV_OVERFLOW=1` is recommended for vLLM because PagedAttention allocs
KV blocks in a dedicated phase that doesn't follow the weight-then-KV timing
the phase detector expects.

### vLLM systemd service

```ini
[Service]
Environment="GREENBOOST_ACTIVE=1"
Environment="GREENBOOST_KV_OVERFLOW=1"
Environment="GREENBOOST_KV_RESERVE_MB=8192"
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
ExecStart=python -m vllm.entrypoints.openai.api_server --model ...
```

---

## Ollama 0.18+ and the CUDA VMM path

Ollama 0.18+ switched its ggml CUDA backend from `cudaMalloc` to the CUDA Virtual Memory
Management API (`cuMemCreate` → `cuMemMap` → `cuMemSetAccess`). GreenBoost intercepts
`cuMemCreate` transparently - no configuration change is needed. When a `cuMemCreate`
call fails due to T1 being full, the shim retries with `CU_MEM_LOCATION_TYPE_HOST`, which
creates a pinned host-memory (system DDR) allocation the GPU accesses over PCIe - the
same effective result as Path B, but initiated from within the VMM code path.

---

## TensorFlow / Keras

TensorFlow loads `libcuda.so` via `ctypes` when `tf.config.list_physical_devices("GPU")`
is called. GreenBoost intercepts from that moment.

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python your_tf_script.py
```

### Memory growth vs pre-allocation

TensorFlow defaults to pre-allocating all GPU memory on startup. GreenBoost reports
the detected T1+T2 total, which TF would try to allocate entirely - set memory growth instead:

```python
import os
os.environ["GREENBOOST_ACTIVE"] = "1"
# LD_PRELOAD must be set before Python starts, not here

import tensorflow as tf

# Prevent TF from trying to allocate the entire T1+T2 virtual pool at once
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# From this point forward cudaMalloc calls are GreenBoost-managed:
# weights → T1, KV-like buffers → T1-priority (if phase heuristic fires)
model = tf.keras.models.load_model("/opt/models/my_model.keras")
```

### Manual KV cache marking for TF custom training

```python
import ctypes, struct, os

def gb_mark_tensor_kv(tf_tensor) -> None:
    """Mark a TensorFlow GPU tensor as KV cache in GreenBoost."""
    # Get raw device pointer via experimental C API
    from tensorflow.python.framework import ops
    ptr = ctypes.cast(
        tf_tensor._handle(),  # internal: raw device pointer
        ctypes.c_void_p
    ).value
    if ptr:
        gb_mark_kv(ptr, tf_tensor.nbytes)
```

### TF with XLA and GreenBoost

XLA compilations may pre-allocate scratch buffers that confuse the phase detector.
Disable the heuristic for XLA workloads:

```bash
GREENBOOST_PHASE_DETECT=0 GREENBOOST_KV_OVERFLOW=0 \
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python xla_training.py
```

---

## KV Cache Fine-Tuning - Bypassing the Phase Heuristic

GreenBoost auto-classifies CUDA allocations via a temporal phase detector
(`INIT → MODEL_LOAD → INFERENCE → STEADY`). This works transparently for Ollama and
llama.cpp, but some engines or custom workloads need explicit control.

### When the heuristic needs help

| Scenario | Symptom | Solution |
|----------|---------|----------|
| ExLlamaV3 (native IOCTL path) | KV allocs don't follow weight-then-KV timing | `GREENBOOST_KV_OVERFLOW=1` |
| Batched inference servers | Multiple parallel KV allocs confuse phase transitions | `GREENBOOST_KV_OVERFLOW=1` |
| Very small models (<8K ctx) | 2 GB default reserve wastes T1 for tiny KV | `GREENBOOST_KV_RESERVE_MB=512` |
| Long-context (128K ctx) | Default 2 GB reserve undersized; KV spills to T2 | `GREENBOOST_KV_RESERVE_MB=6144` |
| Custom training loops | Activation buffers wrongly classified as KV | `GREENBOOST_PHASE_DETECT=0` |
| Verify KV location | Want to confirm KV is in T1, not T2 | Check `kv_t2_mb` in sysfs |

### Environment variable reference

All variables are read by the shim at startup and refreshed every 64 allocations from
the kernel module. They can be set in the Ollama service drop-in, in shell exports, or
prepended to any command.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GREENBOOST_KV_RESERVE_MB` | integer | from kernel (2048) | MB of T1 VRAM reserved exclusively for KV cache. Weights overflow to T2 sooner; KV lands in T1 at 336 GB/s instead of T2 at 32 GB/s. Auto-scaled from `OLLAMA_NUM_CTX` if unset: 8K→1 GB · 32K→2 GB · 64K→4 GB · 128K→6 GB · >128K→8 GB. Capped at T1−2 GB. |
| `GREENBOOST_KV_OVERFLOW` | `0`/`1` | `0` | **Master bypass**: when `1`, every overflow allocation is flagged `GB_ALLOC_KV_CACHE \| GB_ALLOC_T1_PRIORITY` - the phase detector is skipped entirely. Use for ExLlamaV3 or any engine whose overflow allocs are predominantly KV. |
| `GREENBOOST_PHASE_DETECT` | `0`/`1` | `1` | Disable the temporal phase classifier. All overflow allocs use generic flags. Combine with `GREENBOOST_KV_OVERFLOW=1` for full manual control. |
| `GREENBOOST_KV_SIZE_THRESHOLD_MB` | integer | `64` | Minimum alloc size (MB) to classify as KV during the INFERENCE phase. Raise to `256` if activation buffers are being wrongly pinned as KV. |
| `GREENBOOST_VRAM_HEADROOM_MB` | integer | `512` | MB kept free in T1 as a safety buffer. Reduce to `256` to squeeze more weights into T1. |
| `GREENBOOST_VIRTUAL_VRAM_MB` | integer | computed | Override the virtual VRAM total reported to CUDA apps (default: T1 + T2 in MB). |
| `GREENBOOST_ACTIVE` | `0`/`1` | `1` if shim loaded | Must be `1` for the shim to intercept allocations. Set to `0` to disable without unloading. |
| `GREENBOOST_DEBUG` | `0`/`1` | `0` | Enable verbose shim logging to stderr. Each alloc decision is logged: phase, flags, tier placement. |

### Quick recipes

```bash
# Ollama with a 128K context - increase KV reserve to 6 GB
GREENBOOST_KV_RESERVE_MB=6144 ollama run nemotron-3-super:120b

# ExLlamaV3 - bypass phase detector, all overflow is KV
GREENBOOST_KV_OVERFLOW=1 GREENBOOST_ACTIVE=1 \
  LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
  python your_exllama_script.py

# Debug KV placement (stderr shows each allocation decision)
GREENBOOST_DEBUG=1 ollama run glm-4.7-flash:q8_0 2>&1 | grep -E "KV|Phase|VRAM"

# Disable KV heuristic entirely for a custom training script
GREENBOOST_PHASE_DETECT=0 greenboost run python train.py

# Small model, shrink reserve to reclaim 1.5 GB of T1
GREENBOOST_KV_RESERVE_MB=512 ollama run llama3.2:3b
```

### Runtime KV reserve update via IOCTL

The KV reserve can be changed at runtime without restarting Ollama. The shim polls
`GB_IOCTL_GET_INFO` every 64 allocations and picks up the new value:

```bash
# Via Synapse CLI slash command (recommended)
/kv-reserve 4096

# Via greenboost_setup.sh (reads current then patches)
sudo ./greenboost_setup.sh tune-kv-reserve 4096

# Via Python (see synapse/greenboost.py)
from synapse.greenboost import set_kv_reserve
set_kv_reserve(4096)   # returns True on success
```

### `GB_ALLOC_*` flags - programmatic allocation control

These flags are used by ExLlamaV3's `CacheLayer_greenboost` when calling
`GB_IOCTL_ALLOC` directly on `/dev/greenboost`. They bypass the shim entirely and
give the inference engine precise control over tier placement.

```c
/* from greenboost_ioctl.h */
#define GB_ALLOC_WEIGHTS       (1u << 0)  /* model weight tensor             */
#define GB_ALLOC_KV_CACHE      (1u << 1)  /* KV cache - never spills to T3;
                                           * auto-frozen in T2 LRU           */
#define GB_ALLOC_ACTIVATIONS   (1u << 2)  /* ephemeral activation buffer     */
#define GB_ALLOC_FROZEN        (1u << 3)  /* never evict from T2             */
#define GB_ALLOC_NO_HUGEPAGE   (1u << 4)  /* force 4K pages (T3-spillable)   */
#define GB_ALLOC_T1_PRIORITY   (1u << 5)  /* KV-like; moved to LRU head,
                                           * weight bufs evicted first        */
#define GB_ALLOC_KV_COMPRESSED (1u << 6)  /* TurboQuant-compressed KV;
                                           * 3-7× more KV history in T2/T3  */
```

**Flag semantics:**

- `GB_ALLOC_KV_CACHE` - marks the buffer as KV cache. The kernel auto-freezes it in
  the T2 LRU (so it is never evicted to T3) and refuses T3 spill with `ENOSPC`. KV
  bandwidth at T3 (~1.8 GB/s) would collapse generation to unusable speeds.

- `GB_ALLOC_T1_PRIORITY` - combined with `GB_ALLOC_KV_CACHE` it sets `t1_priority=1`
  on the buffer, making weight buffers preferred for eviction over this one. Always
  use together: `GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY`.

- `GB_ALLOC_KV_COMPRESSED` - marks the buffer as TurboQuant-compressed KV. The kernel
  tracks compression savings separately (`kv_compressed_mb`, `kv_compression_bits`
  in pool_info). Set via `GB_IOCTL_SET_TURBOQUANT` first.

- `GB_ALLOC_FROZEN` - prevents eviction from T2 regardless of type. Use for buffers
  that must never be paged out (e.g. LoRA adapter weights during inference).

**Usage pattern (ExLlamaV3 native):**

```c
struct gb_alloc_req req = {
    .size_bytes = kv_size,
    .flags      = GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY,
    .tier_hint  = 1,   /* prefer T1 */
};
if (ioctl(gb_fd, GB_IOCTL_ALLOC, &req) == 0)
    kv_handle = req.handle;
```

### Diagnosing KV spill

If KV cache spills from T1 to T2, the shim logs a warning to stderr and
`kv_t2_mb` in sysfs becomes non-zero:

```bash
# Live monitoring
watch -n1 'grep kv /sys/class/greenboost/greenboost/status'

# One-liner: show KV location
python3 -c "
data = {k.strip(): int(v.strip())
        for line in open('/sys/class/greenboost/greenboost/status')
        if '=' in line for k, v in [line.split('=', 1)]}
kv_t2 = data.get('kv_t2_mb', 0)
kv_used = data.get('kv_used_mb', 0)
print(f'KV used: {kv_used} MB  |  KV in T2: {kv_t2} MB  |  KV in T1: {kv_used - kv_t2} MB')
print('✓ KV fully in T1' if kv_t2 == 0 else f'⚠ KV spilling to T2 - increase GREENBOOST_KV_RESERVE_MB')
"
```

---

## Session Priority Management

For multi-model deployments where several inference processes share T2 DDR, GreenBoost
provides two IOCTLs to control which process's buffers get evicted first when T2 runs low:

| IOCTL | Cmd | Effect |
|-------|-----|--------|
| `GB_IOCTL_SESSION_IDLE` | 18 | Move all caller-PID T2 buffers to LRU tail - first to be evicted under pressure |
| `GB_IOCTL_SESSION_ACTIVE` | 19 | Move all caller-PID T2 buffers to LRU head - last to be evicted |

Call `SESSION_IDLE` when a model goes idle (no pending requests) and `SESSION_ACTIVE` when
a new request arrives. This lets the active model keep its weights in T2 while the idle
model's weights become eviction candidates.

```python
import fcntl, struct, os

# GB_IOCTL_SESSION_IDLE   = _IOW('G', 18, struct gb_session_req)  - 8 bytes
# GB_IOCTL_SESSION_ACTIVE = _IOW('G', 19, struct gb_session_req)  - 8 bytes
_IOC_SESSION_IDLE   = (1 << 30) | (ord('G') << 8) | 18 | (8 << 16)
_IOC_SESSION_ACTIVE = (1 << 30) | (ord('G') << 8) | 19 | (8 << 16)

def gb_session_idle() -> None:
    """Mark this process's T2 buffers as low priority (idle model)."""
    _gb_session_ioctl(_IOC_SESSION_IDLE)

def gb_session_active() -> None:
    """Mark this process's T2 buffers as high priority (active model)."""
    _gb_session_ioctl(_IOC_SESSION_ACTIVE)

def _gb_session_ioctl(cmd: int) -> None:
    dev = "/dev/greenboost"
    if not os.path.exists(dev):
        return
    # struct gb_session_req { u32 pid; u32 reserved; }  - pid=0 means caller's PID
    buf = struct.pack("II", 0, 0)
    fd  = os.open(dev, os.O_RDWR)
    try:
        fcntl.ioctl(fd, cmd, buf)
    finally:
        os.close(fd)
```

---

## Verify GreenBoost is active

```bash
# Should show T2 pool in use (non-zero) after loading a model larger than 12 GB:
cat /sys/class/greenboost/greenboost/status

# Confirm virtual VRAM is visible (should report T1+T2 total, not just physical VRAM):
# From an interactive terminal - GREENBOOST_ACTIVE=1 is already set by profile.d
python -c "
import torch
print('VRAM reported:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# From a non-login shell (cron, Docker, sudo), use the wrapper:
# greenboost run python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

# Check shim is loaded into a running process (e.g. Ollama):
grep greenboost /proc/$(pgrep ollama)/maps | head -3
```

---

## Live debug signals - real-time telemetry during inference

GreenBoost exposes three layers of live telemetry readable at any time, even mid-inference.

### Layer 1: sysfs kernel interface

The kernel module exports a sysfs tree at `/sys/class/greenboost/greenboost/`. These files are always readable while the module is loaded, regardless of whether any app is running.

```bash
# One-liner pool summary - ideal for continuous monitoring
watch -n2 'cat /sys/class/greenboost/greenboost/pool_brief'
# → T1:11GB T2:23/42GB(56%) T3:0/73GB PRESSURE:ok KV_RSV:0MB KV_T2:0MB

# Full tier status with KV cache breakdown and T3 pressure
watch -n2 'cat /sys/class/greenboost/greenboost/status'

# Hardware topology (CPU/GPU/RAM/NVMe, NUMA, kthread affinity) - static
cat /sys/class/greenboost/greenboost/hw_info

# Count of live DMA-BUF tensor handles
watch -n1 'cat /sys/class/greenboost/greenboost/active_buffers'

# KV cache reserve in MB (0 = no reserve, all T2 free for model weights)
cat /sys/class/greenboost/greenboost/kv_reserve_mb
echo 0 | sudo tee /sys/module/greenboost/parameters/kv_reserve_mb   # set live
```

### Layer 2: `/run/greenboost/` runtime files (LD_PRELOAD shim)

These files are written by any process running with `LD_PRELOAD=libgreenboost_cuda.so`. They reflect the *most recent* process that ran with the shim active.

> **Ownership note:** If a previous process (e.g. Ollama running as `ollama`) wrote these files, they will be owned by that user and a new process cannot overwrite them. Fix: `rm -f /run/greenboost/shim_stats /run/greenboost/metrics.json` before starting your process. The `/run/greenboost/` directory is world-writable so any user can unlink these files.

| File | Format | Content |
|------|--------|---------|
| `shim_stats` | `key=value` | Full shim counters - see field table below |
| `metrics.json` | JSON | Same data in JSON with a `feeders[]` array per connected feeder |
| `nvtx_events.log` | TSV | Chronological event trace: allocations, phase transitions, OOM events |
| `phase` | plain text | Current phase: `MODEL_LOAD` / `INFERENCE` / `STEADY` / `OOM_RECOVERY` |

**Key `shim_stats` fields and what they mean:**

```bash
# Watch the most diagnostic fields live during inference
watch -n2 'grep -E "^phase=|^tier_t1_feeder_cur|^tier_t2_feeder_cur|^kernel_dispatch|^remote_alloc_count|^t2_pool_frag|^t2_above_warn|^vram_headroom|^path_[abc]_count" /run/greenboost/shim_stats'
```

| Field | Healthy value | Problem if... |
|-------|--------------|---------------|
| `phase` | `STEADY` or `INFERENCE` | `OOM_RECOVERY` = memory crisis |
| `vram_headroom_mb` | > 200 | < 100 = T1 pressure |
| `tier_t1_feeder_cur_mb` | > 0 when feeder pooled | 0 = feeder not contributing memory |
| `kernel_dispatch_count` | > 0 when feeder pooled | 0 = feeder compute idle despite pooling |
| `remote_alloc_count` | > 0 when feeder pooled | 0 = all allocations staying local |
| `path_a_count` | increases during inference | - (normal T2 DDR path) |
| `path_c_count` | 0 | > 0 = T3 NVMe spill → 10× slowdown |
| `t2_pool_frag_pct` | < 30 | > 80 = fragmentation pressure |
| `t2_above_warn` | 0 | 1 = T2 pool > 75% full |
| `kv_dedup_hits` | increases over time | 0 = KV dedup inactive |

**Inspect feeder connectivity from `metrics.json`:**

```bash
python3 -c "
import json, sys
d = json.load(open('/run/greenboost/metrics.json'))
print(f'Phase:             {d[\"phase\"]}')
print(f'Kernel dispatches: {d[\"kernel_dispatch_count\"]}')
print(f'T1 local cur:      {d[\"tiers\"][\"t1_local\"][\"cur_mb\"]} MB')
print(f'T1 feeder cur:     {d[\"tiers\"][\"t1_feeder\"][\"cur_mb\"]} MB')
print(f'T2 feeder cur:     {d[\"tiers\"][\"t2_feeder\"][\"cur_mb\"]} MB')
for f in d.get('feeders', []):
    print(f'Feeder: {f[\"feeder\"]}  health={f[\"health_state\"]}  bw={f[\"bw_measured_mbs\"]} MiB/s  dispatched={d[\"kernel_dispatch_count\"]}')
"
```

### Layer 3: NVTX event log

The NVTX log is a chronological trace of every significant event the shim handles. It persists across process restarts (appended, not overwritten).

```bash
# Live tail - see allocations and phase transitions as they happen
tail -f /run/greenboost/nvtx_events.log

# Same via greenboost CLI (interactive pager, auto-scrolling)
greenboost nvtx-logs

# Clear log for a fresh diagnostic session
sudo greenboost clear nvtx-logs
```

**Log format:** `<epoch_ns>  <EVENT_TYPE>  <TIER>  <MB>  ptr=<hex>  <message>`

Key event types to look for:

| Event type | Meaning |
|------------|---------|
| `SHIM_INIT` | Process attached to GreenBoost shim |
| `PHASE_MODEL_LOAD` | Model weights loading phase started |
| `PHASE_INFERENCE` | Model loaded, inference phase started |
| `ALLOC_T1_VRAM` | Allocation placed in GPU VRAM (fast path) |
| `ALLOC_T2_POOL` | Allocation placed in T2 DDR pool (path A) |
| `ALLOC_T3_UVM` | Allocation spilled to T3 NVMe (path C) - investigate if frequent |
| `FEEDER_ALLOC` | Allocation placed on a remote feeder GPU |
| `KERNEL_DISPATCH` | CUDA kernel dispatched to feeder for remote execution |
| `OOM_EVICT` | OOM pressure - tensor evicted from T1 to T2/T3 |
| `FREE_T2_POOL` | T2 tensor freed (deallocation) |

### Layer 4: Web dashboard

```bash
# Start greenboost-cli dashboard server (served at http://localhost:7821)
greenboost-cli
# or: python -m greenboost_cli

# Navigate to: http://localhost:7821/greenboost
```

The `/greenboost` page provides:
- **SHIM STATISTICS** - auto-polls `shim_stats` every 2 s: path A/B/C counters, per-tier allocations (local + feeder peaks), kernel dispatch count, T2 fragmentation, T1 headroom
- **FEEDER VITALS** - per-feeder T1/T2/T3 current and peak MB, remote alloc count, connection status badge
- **NVTX EVENT LOG** - scrollable live log of the last N NVTX events with timestamps

### Diagnosing "feeder shows zero activity" (pooled mode)

If `tier_t1_feeder_cur_mb=0` and `kernel_dispatch_count=0` despite `greenboost cluster` showing the feeder connected:

```bash
# 1. Check shim_stats is writable by the current user
ls -la /run/greenboost/shim_stats
# If owned by another user → delete and re-run:
rm -f /run/greenboost/shim_stats /run/greenboost/metrics.json

# 2. Check feeder daemon is actually running on the feeder
ssh ferran@<feeder-ip> 'ss -tlnp | grep 9740'
# If not listening → start it:
ssh ferran@<feeder-ip> 'sudo systemctl start greenboost-netd || sudo nohup /usr/local/bin/greenboost-netd -d -p 9740 &'
# Then re-register:
sudo greenboost connect <feeder-ip>

# 3. Check T3 spillover (NVMe path C) - causes dramatic slowdown
cat /sys/class/greenboost/greenboost/pool_brief
# If T3 > 0 during active inference:
echo 0 | sudo tee /sys/module/greenboost/parameters/kv_reserve_mb

# 4. One-command feeder redeploy if daemon is stale
sudo greenboost feeders upgrade-greenboost
```

---

## TurboQuant - global toggle (v2.9+)

TurboQuant compresses the K/V cache during attention computation, reducing the bandwidth needed to move tensors between T2 DDR and T1 VRAM.

### Enable / disable system-wide

```bash
# Enable for Ollama (q4_0 KV cache) + Python inference apps
sudo greenboost turboquant on

# Disable and revert to q8_0 KV cache
sudo greenboost turboquant off

# Check current state
greenboost turboquant status
```

`turboquant on` does two things simultaneously:

1. **Ollama** - updates `/etc/systemd/system/ollama.service.d/99-greenboost.conf` to set `OLLAMA_KV_CACHE_TYPE=q4_0` and `OLLAMA_FLASH_ATTENTION=1`, then restarts the Ollama service.
2. **Python apps** - writes `/etc/greenboost/turboquant.enabled` and creates `/etc/profile.d/greenboost-turboquant.sh` which exports `GREENBOOST_TURBOQUANT=1` for all future login shells.

### Using TurboQuant in Python code

When `GREENBOOST_TURBOQUANT=1` is set (or the flag file exists), Python inference scripts can opt in:

```python
import sys, os
sys.path.insert(0, "/path/to/greenboost_all/greenboost")
from gb_attn import turboquant_attention

# Recommended: asymmetric K=4bit / V=3bit (~4× bandwidth reduction, +0.23% PPL)
with turboquant_attention(k_bits=4, v_bits=3, sparse_v=True):
    output = model(input)
```

Or check the flag and apply conditionally:

```python
from pathlib import Path
if Path("/etc/greenboost/turboquant.enabled").exists() \
        or os.environ.get("GREENBOOST_TURBOQUANT") == "1":
    from gb_attn import turboquant_attention
    ctx = turboquant_attention(k_bits=4, v_bits=3)
else:
    from contextlib import nullcontext
    ctx = nullcontext()

with ctx:
    output = model(input)
```

---

## Optimized inference configuration generator (v2.9+)

`greenboost gen-inference-config` emits a configuration block tuned for the current machine and environment. Useful when setting up Ollama or a Python inference service on a machine that may be a VM, container, or WSL2 instance.

```bash
# Print Ollama env block + Python snippet to stdout
greenboost gen-inference-config --format both

# Save to a file, with TurboQuant explicitly enabled
greenboost gen-inference-config --format ollama \
    --turboquant \
    --output /etc/greenboost/inference.env

# Then source it before starting Ollama manually
export $(grep -v '^#' /etc/greenboost/inference.env | xargs)
ollama serve
```

Flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--format ollama\|hf\|both` | `ollama` | Which config blocks to emit |
| `--turboquant` | (reads flag file) | Force TurboQuant settings on |
| `--no-turboquant` | - | Force TurboQuant settings off |
| `--output <path>` | stdout | Write to file instead of printing |
| `--model <name>` | - | Include a model name hint in the output |

---

## Cluster - connect remote machines as GPU memory and compute feeders

GreenBoost can use the VRAM, RAM, and **GPU compute** of other machines on your network. The host machine (running Ollama) aggregates all feeder resources into a single virtual GPU. Model layers that overflow into a feeder's VRAM are not just stored there - they are also computed there. GreenBoost automatically dispatches CUDA kernels to whichever machine owns the data, with no manual configuration.

### Set up a feeder machine

```bash
# On the feeder machine - start the daemon
sudo greenboost feed start

# The daemon listens on TCP port 9740
```

### Connect from the host

```bash
# Connect to a feeder (saves to /etc/greenboost/cluster.conf)
sudo greenboost connect 192.168.1.50

# Connect to multiple feeders
sudo greenboost connect 192.168.1.51
sudo greenboost connect 192.168.1.52

# View live cluster status - updates every 5 seconds
greenboost cluster

# Disconnect a feeder
sudo greenboost disconnect 192.168.1.50
```

### What `greenboost cluster` shows

`greenboost cluster` is the live interactive view of all connected feeders. It displays:
- Each feeder's IP, VRAM available, DDR available, connection latency
- Total memory contributed by the cluster
- Per-tier breakdown (T1 feeder / T2 feeder / T3 feeder)

Press `Ctrl+S` to refresh immediately, `Ctrl+C` to exit.

### Upgrading GreenBoost on feeders

After building or updating GreenBoost on the host, push the new binaries to all feeders in one command:

```bash
# Push greenboost-netd + CUDA shim + setup script + build stamp to all feeders,
# then restart their daemon automatically.
sudo greenboost feeders upgrade-greenboost

# Check that all machines are on the same build:
greenboost built-stamp --feeders
```

`built-stamp` shows the `BUILD_ID`, version, git hash, and build date for the local machine.
With `--feeders` it also SSHes to every configured feeder and prints their stamp side-by-side.
Mismatched stamps are highlighted in amber as a reminder to upgrade.

```bash
# Check only the local stamp:
greenboost built-stamp
```

### How the cluster works with Ollama

GreenBoost presents **one single virtual GPU** to Ollama, regardless of how many feeders are connected. Ollama sees one large GPU with total memory = local T1+T2+T3 + all feeder T1+T2+T3. All memory is pooled into device 0 - there is no multi-GPU scheduling.

**Memory allocation order** when VRAM overflows:
```
Local T1 VRAM → Feeder T1 VRAM → Local T2 DDR → Feeder T2 DDR → Local T3 NVMe → Feeder T3 NVMe
```

**Compute dispatch (data-driven, automatic):**
When a CUDA kernel is launched, GreenBoost inspects the kernel's arguments. If any tensor lives on a feeder (identified by its fake pointer in the `0xAA00…` range), the entire kernel is sent to that feeder's GPU over TCP to be executed there. The result is returned to the host. This means feeder GPUs are not idle storage - they actively run inference for the layers stored in their VRAM. No configuration is needed; dispatch follows the data automatically.

---

## Synapse CLI - terminal AI assistant with GreenBoost integration (v2.9+)

Synapse CLI is the companion terminal assistant at `greenboost_synapse_cli_new/`. Install it once, then run `synapse-cli` or `python synapse_cli.py`.

### GreenBoost features inside Synapse

**Startup banner** shows live GreenBoost state:
```
🐙 Synapse CLI
──────────────────────────────────────────────────
  Model        nutboy02/Qwen3.6-35B-...  (ollama)
  Permissions  auto
  GreenBoost   GB  TQ  +1 feeder(s)
  /backend · /model · /turboquant · /help
```
`GB` = GreenBoost installed · `TQ` = TurboQuant on · `+N feeder(s)` = cluster connected

**`/turboquant` slash command:**
```
/turboquant status    show current TurboQuant state
/turboquant on        sudo greenboost turboquant on
/turboquant off       sudo greenboost turboquant off
```

**System prompt context** - every AI request includes a short GreenBoost paragraph so the model knows it has extended memory available and can be asked to work with large contexts.

### Running the Qwen3.6-35B model

```bash
cd greenboost_synapse_cli_new
python synapse_cli.py --model "qwen3.6:latest"
```

This routes through the Ollama backend (`localhost:11434`). GreenBoost provides the extended memory (T2+T3+cluster) so the 35B model fits even when local VRAM is smaller than the model size. Enable TurboQuant before loading for best throughput:

```bash
sudo greenboost turboquant on
python synapse_cli.py --model "qwen3.6:latest"
```

---

## Diffusion Models - Best Practices

FLUX.1-dev and FLUX.2-klein (image generation) have different memory and performance characteristics from LLM models. Key recommendations for running diffusion pipelines through GreenBoost:

### Model selection and speed

| Model | Mode | VRAM | Est. speed | GreenBoost required |
|---|---|---|---|---|
| FLUX.2-klein | FP8 + TurboQuant | ~16 GB T1+T2 | **4–6 s/image** ★ | Optional |
| FLUX.2-klein | BF16 + TurboQuant | ~22 GB T1+T2 | 5–8 s/image | Required |
| FLUX.1-dev | NF4 + TurboQuant | ~8 GB T1 | ~2 min/image | Optional |
| FLUX.1-dev | BF16 + TurboQuant | ~23 GB T1+T2 | ~4 min/image | Required |

★ **Recommended default**  art pipeline;

### Environment variables for diffusion pipelines

```bash
# Required for GreenBoost aggregation
export GREENBOOST_ACTIVE=1
export LD_PRELOAD='/usr/local/lib/libgreenboost_cuda.so'

# Diffusion models have no KV cache - set to 0 to give all T2 to model weights
export GREENBOOST_KV_RESERVE_MB=0

# Enable T1↔T2 bandwidth compression (reduces PCIe congestion during denoising steps)
export GREENBOOST_KV_COMPRESS=1

# Keep Path A0 (DMA-BUF zero-copy) disabled - cudaImportExternalMemory not
# supported on RTX 5070 Laptop + current driver combination
export GREENBOOST_A0_DISABLE=1
```

### Why KV reserve = 0 for diffusion

Unlike LLMs, diffusion transformers perform attention over spatial latent tokens - there is no autoregressive KV cache to accumulate across denoising steps. Each step is a full forward pass with no persistent KV state. Setting `GREENBOOST_KV_RESERVE_MB=0` dedicates all T2 DDR to model weight storage, which is where diffusion models need the memory.

### TurboQuant with diffusion models

TurboQuant compresses attention tensors during the forward pass. For diffusion models:
- Effective for BF16 modes: large per-step attention tensors benefit from 4-bit K / 3-bit V compression
- Less critical for FP8 modes: FP8 weights are already compact; TurboQuant still helps on T1↔T2 transfers
- Enable with `--turboquant` flag in `gen_art.py` / `gen_manual_art.py`

### T3 spillover detection

T3 (system RAM via PCIe) causes dramatic slowdown (~10× slower than T2). Monitor during generation:

```bash
watch -n2 'cat /sys/class/greenboost/greenboost/status'
# T3 used_mb should stay at 0 during normal klein/flux generation
```

If T3 > 0 appears, the model did not fit in T1+T2. Switch to a smaller quantization mode or reduce batch size.

---


## Debug Vitals

GreenBoost provides a comprehensive, toggleable debug vitals system that surfaces
**9 data sources** in a single command. Essential for diagnosing issues, measuring
performance, and guiding enhancement work.

### Quick start

```bash
# Show full vitals dump immediately (no flag needed)
greenboost debug vitals

# Enable persistent vitals mode - activates verbose instrumentation
sudo greenboost debug vitals on

# Disable persistent vitals - restores minimal-overhead settings
sudo greenboost debug vitals off
```

### What vitals on/off does

**`sudo greenboost debug vitals on`:**
1. Sets kernel `debug_mode=1` via sysfs (live, no restart required)
2. Writes `/etc/profile.d/greenboost-vitals.sh` → exports `GREENBOOST_NVTX_VERBOSE=1`
3. Creates `/etc/greenboost/debug_vitals.enabled` flag file
4. Restarts `greenboost-idle-reclaim` daemon
5. Lists other GB-aware services that need manual restart
6. Warns if a reboot is needed for full shim injection

**`sudo greenboost debug vitals off`:**
1. Resets `debug_mode=0`
2. Removes profile.d file
3. Removes flag file and restarts services

### Data sources table

| Source | Path | What it provides |
|--------|------|-----------------|
| IOCTL | `/dev/greenboost` | T1/T2/T3/KV/pressure, 24 fields real-time |
| Sysfs brief | `/sys/class/greenboost/greenboost/pool_brief` | One-liner T1/T2/T3 for polling |
| Sysfs status | `/sys/class/greenboost/greenboost/status` | Full text status, KV placement |
| Shim stats | `/run/greenboost/shim_stats` | Path counters, H2D/D2H, dispatch, frag, dedup |
| Phase file | `/run/greenboost/phase` | Current CUDA phase string |
| NVTX log | `/run/greenboost/nvtx_events.log` | Per-allocation events with timestamps |
| Metrics JSON | `/run/greenboost/metrics.json` | Structured JSON + feeder vitals |
| nvidia-smi ext | nvidia-smi extended query | Temp, power, util%, SM/mem clocks |
| Kernel journal | `journalctl -k` | Module events (when debug_mode=1) |

### From greenboost-cli

```bash
/gb-vitals          # Full vitals in the Python CLI (Rich rendering)
/gb-vitals on       # Enable (calls sudo greenboost debug vitals on)
/gb-vitals off      # Disable
```

### In scripts

Scripts and wizards automatically show the extended vitals panel when the flag file exists:

```bash
# art_wizard.sh: extended panel after T1/T2/T3 bars when vitals is ON
# factory.sh:    greenboost status shown before/after each stage
# gen_art.py:    reads shim_stats at generation checkpoints
```

### Typical use cases

| Scenario | What to look for |
|----------|-----------------|
| T3 spillover | `t3_used_mb > 0` or Path C count increasing |
| KV cache pressure | `kv_t2_mb > 0` means KV spilling into DDR |
| Memory fragmentation | `shim_frag_pct > 30` indicates T2 fragmentation |
| Feeder not being used | `shim_remote_alloc_count == 0` despite feeder connected |
| Phase stuck | `shim_phase` stays `MODEL_LOAD` after model should be ready |
| OOM guard triggered | `oom_active = true` in IOCTL data |
| Kernel events | `journalctl -k --grep=greenboost` when debug_mode=1 |

---

## Long-running stability monitor (v2.10+)

Once you have a workload running for hours or days, you want a sanity
check that nothing is drifting. That's what `greenboost stability` does.

### What it watches

The script (`gb_stability_monitor.py`) polls
`/run/greenboost/shim_stats` at a configurable interval and flags four
invariant violations:

1. **Counter regressed.** Fields like `kernel_dispatch_count`,
   `h2d_mb`, `d2h_mb`, `remote_alloc_count`, `tier_*_lifetime_mb`
   should only ever go up. If one decreases without an accompanying
   PID change, that's either a wraparound bug or a reset that wasn't
   declared.
2. **Tier-gauge leak.** Gauges like `kv_t1_tracked_mb`,
   `tier_t1_local_cur_mb`, `tier_t2_local_cur_mb` should drop close to
   zero once the shim's reported phase leaves `INFERENCE` or `STEADY`
   and a cool-down period passes. If the gauge stays above 10 % of its
   peak after 2 minutes of idle, that's a memory leak somewhere.
3. **Fragmentation drift.** `t2_pool_frag_pct` and
   `kv_internal_frag_mb` climbing monotonically across 30 consecutive
   samples (≈15 min at the default 30-second interval) without ever
   retreating is reported as drift.
4. **Shim wedged.** If `timestamp` in `shim_stats` doesn't move for 90
   seconds while the file still exists, the shim has stopped writing -
   usually a sign of a hang inside an LD_PRELOAD hook.

A shim PID change resets the monotone baselines automatically - it's
not a leak, just a fresh process.

### Running it

```bash
# Short check - useful in CI:
greenboost stability --interval 5 --duration 60 --strict

# Day-long observation, JSON output for log shippers:
greenboost stability --interval 30 --duration 86400 --json \
    --log /var/log/greenboost-stability.log

# Until you Ctrl-C, plain text to stdout:
greenboost stability
```

Exit codes:
- `0` clean run, or `--strict` not used.
- `1` at least one violation observed (only when `--strict`).
- `2` invalid arguments / setup error.

### Use with Prometheus

If you already scrape `greenboost_exporter.py`, the stability monitor
covers a different angle: the exporter tells you *what the system
currently looks like*, the stability monitor tells you *whether that
view has been internally consistent over time*.

A workflow that has worked in practice: scrape metrics into Prometheus
for short-term graphs and alerts, run the stability monitor with
`--json --log /var/log/greenboost-stability.log` for the long-running
invariant trail.

---

## Worker pool (v2.10+, opt-in)

`greenboost-netd` is single-threaded by default. A slow CUDA op on one
client (a big `cudaMalloc`, a large `cudaMemcpy`) blocks the reactor
and every other client waits behind it. For most setups that's fine -
clusters are usually 1–4 feeders serving one host.

If you're running a larger cluster or a multi-tenant feeder, set:

```bash
GREENBOOST_WORKERS=8 sudo greenboost feed start
```

That spawns up to 32 worker threads behind a FIFO queue of 256 jobs.
When the queue is full, `gb_workpool_submit()` returns `-1` and the
caller falls back to inline execution - no work is dropped silently.

At the time of v2.10 the scaffold is in place but only diagnostic
handlers use it. Migration of `handle_cuda_memcpy_h2d/d2h` and
`handle_cuda_exec` onto the pool is gated on a lock-discipline audit
(several globals - `g_alloc_lock`, `g_inflight_ops`, `ra_count` -
currently assume single-threaded callers).

You can verify the pool is alive by watching netd logs for
`PR-VV: worker pool active (N threads)` at startup and the drain line
at shutdown reporting submitted/completed/rejected counts.

---

## Sister project: GreenBoost Gaming Suite

Through v2.9 the main GreenBoost repository shipped an experimental
Vulkan layer (`libVkLayer_greenboost.so`) that let Steam games via
Proton see the GreenBoost memory pool. That layer touched a different
graphics API stack (Vulkan, not CUDA), required different testing, and
attracted a different audience (gamers vs. ML engineers).

In v2.10 the Vulkan layer was **extracted** into its own repository:

> **[GreenBoost Gaming Suite](../greenboost_gaming/)**

What you'll find there:

- The **GVM Vulkan implicit layer** that inflates each game's reported
  device-local heap by routing overflow allocations through the
  GreenBoost CUDA pool. Vulkan apps see VRAM that doesn't fit on the
  card.
- A **GTK4 desktop application** (shows up in the GNOME app grid) that
  scans your Steam library, lists installed games with their current
  Proton/DLSS versions, and applies per-game optimal settings.
- A **DLSS / FSR / XeSS updater** that finds the relevant DLLs inside
  each Steam Proton prefix and bumps them to the latest version.
- A **GPU profile editor** (clocks, power limit, fan curve) for
  sustained gaming sessions.
- The **GreenBoost Proton wrapper** that injects the environment
  variables games need to activate the layer.

### Installation pre-req

The Gaming Suite **requires GreenBoost (this project) to be installed
first**. The Vulkan layer is just a frontend onto the same CUDA pool
that your inference workflow already uses, so the kernel module and
shim must be in place. The Gaming Suite installer detects whether
GreenBoost is present and refuses with a clear instruction if not.

### Do I need it?

- Doing AI inference only → **no.** You can ignore the Gaming Suite
  entirely.
- Mostly gaming, no AI work → install GreenBoost (this repo) **and**
  the Gaming Suite. The CUDA inference parts of GreenBoost stay idle;
  it's the kernel module + memory pool that the Vulkan layer needs.
- Both → install both. GreenBoost provides the pool, the Gaming Suite
  exposes it to Vulkan games, and your inference workflow continues
  to work unchanged.

---
