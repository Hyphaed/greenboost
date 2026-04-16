# GreenBoost — integration guide for inference tools

Check [GREENBOOST_COMMANDS.md](GREENBOOST_COMMANDS.md) for all the available commands
Check [ARCHITECTURE.md](ARCHITECTURE.md) to know about greenboost architecture

## How GreenBoost hooks in

GreenBoost works by intercepting `cudaMalloc` and related CUDA symbols via `LD_PRELOAD`, plus intercepting `dlsym` so that the virtual VRAM total (T1 VRAM + T2 DDR pool, computed from detected hardware) is returned to any app that queries VRAM size at runtime through `dlopen`+`dlsym`.

The shim loads system-wide via `/etc/ld.so.preload` but stays **inert** until
`GREENBOOST_ACTIVE=1` is set. In **interactive login shells** (terminal, SSH),
`/etc/profile.d/greenboost.sh` exports `GREENBOOST_ACTIVE=1` automatically — no wrapper
needed. Just run your script directly:

```bash
python your_script.py               # shim already active in login shells
python -m vllm.entrypoints.openai.api_server ...
```

In **non-login contexts** (cron jobs, Docker entrypoints, sudo scripts, spawned subshells)
profile.d is not sourced. Use the wrapper or set the variable explicitly:

```bash
# Wrapper — sets GREENBOOST_ACTIVE=1 for one command
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

TGI uses a PyTorch backend — same lazy-CUDA pattern:

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

Transformers loads CUDA through PyTorch — same lazy-CUDA pattern as vLLM and TGI.

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
    device_map="auto",          # sees T1+T2 total — loads entire model onto T1+T2
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
tensors — `GREENBOOST_ACTIVE=1` is still needed for virtual VRAM reporting.

---

## PyTorch — direct tensor allocation control

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

# GB_IOCTL_MADVISE = _IOW('G', 4, struct gb_madvise_req)  — 8 bytes
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

# 1. Load model weights (phase = MODEL_LOAD — reserve inactive, weights fill T1)
model = MyModel().cuda()
model.load_state_dict(torch.load("model.pt", map_location="cuda"))

# 2. Allocate KV cache (phase = INFERENCE — reserve activates, KV lands in T1)
kv_cache = torch.zeros(batch, n_layers, seq_len, d_model, device="cuda")
# ↑ GreenBoost auto-classifies this as KV via quiet-gap heuristic
#   OR call gb_mark_kv() above for certainty

# 3. Run inference
output = model.generate(input_ids, past_key_values=kv_cache)
```

If your script loads KV cache before or interleaved with weights, set
`GREENBOOST_KV_OVERFLOW=1` to disable the heuristic.

---

## Hugging Face Transformers — direct KV cache control

Transformers uses PyTorch's CUDA allocator; GreenBoost intercepts all `cudaMalloc`
calls. `device_map="auto"` queries VRAM size — GreenBoost reports the detected T1+T2
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
    device_map="auto",          # sees T1+T2 total — all layers on GPU
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

## vLLM — direct KV cache placement

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
vLLM will try to allocate ~11.4 GB of KV cache from "GPU" memory — GreenBoost
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
`cuMemCreate` transparently — no configuration change is needed. When a `cuMemCreate`
call fails due to T1 being full, the shim retries with `CU_MEM_LOCATION_TYPE_HOST`, which
creates a pinned host-memory (system DDR) allocation the GPU accesses over PCIe — the
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
the detected T1+T2 total, which TF would try to allocate entirely — set memory growth instead:

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

## KV Cache Fine-Tuning — Bypassing the Phase Heuristic

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
| `GREENBOOST_KV_OVERFLOW` | `0`/`1` | `0` | **Master bypass**: when `1`, every overflow allocation is flagged `GB_ALLOC_KV_CACHE \| GB_ALLOC_T1_PRIORITY` — the phase detector is skipped entirely. Use for ExLlamaV3 or any engine whose overflow allocs are predominantly KV. |
| `GREENBOOST_PHASE_DETECT` | `0`/`1` | `1` | Disable the temporal phase classifier. All overflow allocs use generic flags. Combine with `GREENBOOST_KV_OVERFLOW=1` for full manual control. |
| `GREENBOOST_KV_SIZE_THRESHOLD_MB` | integer | `64` | Minimum alloc size (MB) to classify as KV during the INFERENCE phase. Raise to `256` if activation buffers are being wrongly pinned as KV. |
| `GREENBOOST_VRAM_HEADROOM_MB` | integer | `512` | MB kept free in T1 as a safety buffer. Reduce to `256` to squeeze more weights into T1. |
| `GREENBOOST_VIRTUAL_VRAM_MB` | integer | computed | Override the virtual VRAM total reported to CUDA apps (default: T1 + T2 in MB). |
| `GREENBOOST_ACTIVE` | `0`/`1` | `1` if shim loaded | Must be `1` for the shim to intercept allocations. Set to `0` to disable without unloading. |
| `GREENBOOST_DEBUG` | `0`/`1` | `0` | Enable verbose shim logging to stderr. Each alloc decision is logged: phase, flags, tier placement. |

### Quick recipes

```bash
# Ollama with a 128K context — increase KV reserve to 6 GB
GREENBOOST_KV_RESERVE_MB=6144 ollama run nemotron-3-super:120b

# ExLlamaV3 — bypass phase detector, all overflow is KV
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

### `GB_ALLOC_*` flags — programmatic allocation control

These flags are used by ExLlamaV3's `CacheLayer_greenboost` when calling
`GB_IOCTL_ALLOC` directly on `/dev/greenboost`. They bypass the shim entirely and
give the inference engine precise control over tier placement.

```c
/* from greenboost_ioctl.h */
#define GB_ALLOC_WEIGHTS       (1u << 0)  /* model weight tensor             */
#define GB_ALLOC_KV_CACHE      (1u << 1)  /* KV cache — never spills to T3;
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

- `GB_ALLOC_KV_CACHE` — marks the buffer as KV cache. The kernel auto-freezes it in
  the T2 LRU (so it is never evicted to T3) and refuses T3 spill with `ENOSPC`. KV
  bandwidth at T3 (~1.8 GB/s) would collapse generation to unusable speeds.

- `GB_ALLOC_T1_PRIORITY` — combined with `GB_ALLOC_KV_CACHE` it sets `t1_priority=1`
  on the buffer, making weight buffers preferred for eviction over this one. Always
  use together: `GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY`.

- `GB_ALLOC_KV_COMPRESSED` — marks the buffer as TurboQuant-compressed KV. The kernel
  tracks compression savings separately (`kv_compressed_mb`, `kv_compression_bits`
  in pool_info). Set via `GB_IOCTL_SET_TURBOQUANT` first.

- `GB_ALLOC_FROZEN` — prevents eviction from T2 regardless of type. Use for buffers
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
print('✓ KV fully in T1' if kv_t2 == 0 else f'⚠ KV spilling to T2 — increase GREENBOOST_KV_RESERVE_MB')
"
```

---

## Session Priority Management

For multi-model deployments where several inference processes share T2 DDR, GreenBoost
provides two IOCTLs to control which process's buffers get evicted first when T2 runs low:

| IOCTL | Cmd | Effect |
|-------|-----|--------|
| `GB_IOCTL_SESSION_IDLE` | 18 | Move all caller-PID T2 buffers to LRU tail — first to be evicted under pressure |
| `GB_IOCTL_SESSION_ACTIVE` | 19 | Move all caller-PID T2 buffers to LRU head — last to be evicted |

Call `SESSION_IDLE` when a model goes idle (no pending requests) and `SESSION_ACTIVE` when
a new request arrives. This lets the active model keep its weights in T2 while the idle
model's weights become eviction candidates.

```python
import fcntl, struct, os

# GB_IOCTL_SESSION_IDLE   = _IOW('G', 18, struct gb_session_req)  — 8 bytes
# GB_IOCTL_SESSION_ACTIVE = _IOW('G', 19, struct gb_session_req)  — 8 bytes
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
    # struct gb_session_req { u32 pid; u32 reserved; }  — pid=0 means caller's PID
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
# From an interactive terminal — GREENBOOST_ACTIVE=1 is already set by profile.d
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
