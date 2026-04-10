# GreenBoost — Library Patches

These are the GreenBoost-specific modifications to third-party libraries.
Apply them after cloning the upstream library into `libraries/`.

## ExLlamaV3 — GreenBoost KV Cache Layer

**Files:**
- `exllamav3/exllamav3/cache/greenboost.py` — `CacheLayer_greenboost` and `GreenBoostCache`
- `exllamav3/exllamav3/cache/__init__.py` — exports `CacheLayer_greenboost`, `GreenBoostCache`

**Apply:**
```bash
# Clone upstream ExLlamaV3 into libraries/:
git clone https://github.com/turboderp-org/exllamav3 libraries/exllamav3

# Apply patches:
cp patches/exllamav3/exllamav3/cache/greenboost.py libraries/exllamav3/exllamav3/cache/
cp patches/exllamav3/exllamav3/cache/__init__.py    libraries/exllamav3/exllamav3/cache/

# Install into GreenBoost venv:
STLOADER_USE_URING=1 /opt/greenboost/venv/bin/pip install -e libraries/exllamav3 --no-build-isolation
```

**What the patch adds:**
- `CacheLayer_greenboost`: allocates KV cache via `/dev/greenboost` IOCTL (DMA-BUF pages)
- Zero-copy tensor bridge: `mmap(dma_buf_fd) → np.frombuffer() → torch.from_numpy()`
- `GreenBoostCache`: convenience factory that opens one `/dev/greenboost` fd for all layers

---

## Other Libraries (no patches required)

These are used as-is from their upstream sources:

| Library | URL | Purpose |
|---------|-----|---------|
| kvcompress / kvpress | https://github.com/westers/kvcompress | Runtime KV cache compression |
| NVIDIA ModelOpt | https://github.com/NVIDIA/Model-Optimizer | Post-training quantization (FP8, INT4-AWQ) |
| TensorRT-Edge-LLM | https://github.com/NVIDIA/TensorRT-Edge-LLM | TRT engine export |
| LoRA (loralib) | https://github.com/microsoft/LoRA | LoRA layers (used by Unsloth) |

The `greenboost_setup.sh full-install` command clones and installs all of these automatically.
