#!/usr/bin/env bash
# Feeder-residency test , drives the SAME engine ai-forge's LLM stage uses
# (ollama's llama-server + the qwen36 22 GB model) with conditions that force
# feeder GPU VRAM, feeder DDR, and feeder GPU compute:
#   * a VRAM hog pins ~8 GB locally so KV/compute buffers overflow,
#   * GREENBOOST_HOST_RAM_FLOOR_MB=48000 makes local T2 refuse the weights,
#     so the 21.7 GB buffer lands on the feeder's DDR (T2_feeder),
#   * kernels touching feeder-resident args dispatch via cudaLaunchKernelExC.
#
# Run from the repo root after `make shim` (uses the REPO build, no root):
#   tests/cluster/run_feeder_residency_test.sh
#
# Expected today (2026-07-06): 22101 H2D chunks land on the feeder
# (watch `free -g` on the feeder grow by ~22 GB), then the warmup dispatches
# rms_norm_f32 to the feeder and aborts with "illegal memory access" ,
# the known mixed-args wall (local-VRAM activation pointers inside a
# remotely-dispatched kernel). Fixing that = populate the exec message's
# upload/download descriptors for local args. See workflow/known-issues.md.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${TMPDIR:-/tmp}/gb_feeder_test"
mkdir -p "$S"
MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d
HOGPY="${GB_HOG_PYTHON:-$HOME/Dev/ai-forge/ComfyUI/venv/bin/python}"

echo "[test] starting 8 GiB VRAM hog..."
GREENBOOST_ACTIVE=0 "$HOGPY" -c "
import torch, time
hog = [torch.empty(1<<28, dtype=torch.float32, device='cuda') for _ in range(8)]
print('HOG ready', flush=True); time.sleep(1800)" > "$S/hog.log" 2>&1 &
HOG_PID=$!
trap 'kill $HOG_PID 2>/dev/null || true' EXIT
until grep -q 'HOG ready' "$S/hog.log" 2>/dev/null; do sleep 1; done

echo "[test] launching llama-server with repo shim (weights → feeder DDR)..."
GREENBOOST_ACTIVE=1 \
LD_PRELOAD=/usr/local/lib/libgreenboost_vmm_override.so:$REPO/libgreenboost_cuda.so \
GREENBOOST_DEBUG=1 GREENBOOST_NET_DEBUG=1 \
GREENBOOST_HOST_RAM_FLOOR_MB=48000 \
GREENBOOST_VIRTUAL_VRAM_MB=43008 GREENBOOST_KV_RESERVE_MB=0 \
GGML_CUDA_DISABLE_GRAPHS=1 GREENBOOST_FORCE_CC_MAJOR=12 \
GGML_BACKEND_PATH=/usr/local/lib/ollama/cuda_v13/libggml-cuda.so \
LD_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v13 \
timeout 900 /usr/local/lib/ollama/llama-server \
  --model "$MODEL" --port 8113 --host 127.0.0.1 --no-webui --offline \
  -c 16384 -np 1 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on -b 512 -ub 512 2>&1 | tee "$S/server.log" | \
  grep --line-buffered -E 'cudaMalloc feeder|data-driven dispatch|model loaded|CUDA error|RAM floor'

echo "[test] summary:"
echo "  feeder T2 allocs : $(grep -c 'cudaMalloc feeder T2' "$S/server.log" || true)"
echo "  H2D transfers    : $(grep -c 'memcpy H2D' "$S/server.log" || true)"
echo "  remote dispatches: $(grep -c 'data-driven dispatch' "$S/server.log" || true)"
