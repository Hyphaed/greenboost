# Running Qwen3.6-27B-Fable-Fusion on a two-GPU cluster, and why it stays host-only

**Audience:** an AI engineering student who knows what a transformer is, has run
Ollama, and wants to understand what GreenBoost actually *does* when a 17 GiB
model answers on a 12 GB graphics card, and why "add a second GPU" didn't turn
out to be the win it sounds like.

This is a walkthrough of one real configuration, end to end, including the
dead end:

| | |
|---|---|
| Model | `DavidAU/Qwen3.6-27B-Fable-Fusion-711-...-NEO-MAX-MTP-GGUF` (HF GGUF, `MTP-Q4_K_M`, **17.2 GiB**) |
| Architecture | `qwen35` — **dense**, 27B, every parameter fires on every token. Hybrid: standard attention layers plus **Gated Delta Net** (linear-attention) layers |
| Context | 262,144 tokens native; the box's VRAM currently clamps served context to 2,048 |
| Host | RTX 5070, **12 GB** VRAM · 61 GB DDR5, i9-14900KF (8P+16E) |
| Feeder | RTX 5070 Laptop, **8 GB** VRAM · 29 GB DDR5, over a 1 Gbps LAN |
| Served by | GB-Synapse → `llama-server`, **host-only**, on `:11435` |

The whole problem in one line: **17.2 GiB of weights, ~10-12 GB of usable
single-GPU VRAM.** Unlike the cluster's previous reference model, this one is
**dense, not MoE** — that single fact changes almost every decision below.

---

## 1. Dense changes the placement problem — there is no "cheap tensor" to evict

The previous reference workload was a 256-expert MoE: 92% of its bytes were
experts that only fired 8-per-token, so the fix was surgical — move the
*experts*, keep the *attention*, and pay almost nothing for it
(`--n-cpu-moe`). That trick doesn't exist here. In a dense model every
tensor in every layer fires on every token, so there is no low-traffic
subset to evict cheaply. The only lever left is coarser: **how many whole
layers** sit on the GPU (`-ngl N`) versus the CPU. Every layer moved off the
GPU costs its full share of compute, not just its bytes.

```
weights=17.2GB/10.0GB budget → PARTIAL CPU OFFLOAD (slow): 28/65 layers on GPU
```

This is the honest, structural reason this model is slower on this hardware
than the old MoE reference was: a dense model exceeding VRAM has no free
lunch left to take.

---

## 2. A real bug this model exposed: the compute-graph reserve was too small

The very first load attempt failed outright:

```
resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated Delta Net
  (chunked) is assigned to device CUDA0 (usually due to missing support)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 183.73 MiB on device 0:
  cudaMalloc failed: out of memory
```

GreenBoost's own layer-count planner (`_fit_gpu_layers`) sized `-ngl` against
a flat, %-derived compute-graph reserve tuned for plain attention-only
architectures. This model's hybrid **Gated Delta Net** layers need more
workspace than that flat estimate provides — the weights allocation
succeeded, but a 183 MB compute buffer on top of it didn't. Real evidence,
not a guess: fixed by retrying once at a smaller layer count when this
specific OOM shape is seen (`gb_synapse_backends.py`). Live-verified twice:
both OOM'd at `-ngl 25`, retried and served successfully at `-ngl 20`.

> **Lesson:** a placement budget tuned against one architecture's shape can
> be wrong for a different one, even on the same hardware, even at the same
> nominal VRAM number. The fix is a fast, cheap retry with a real failure
> signature to key off, not a bigger fudge factor guessed in advance.

---

## 3. The RPC-cluster attempt, and the two real bugs it found

The obvious next move: bring the feeder's 8 GB online via `--rpc` and split
the 65 layers across both cards. This surfaced two more real bugs before it
worked at all, and a third reason it isn't used in production even now that
it does.

**Bug: a stale feeder engine binary.** `omen`'s `rpc-server` was 15 days
older than the host's and had no `PINNED_COMMIT` in its vendored
`llama.cpp` checkout — old enough to predate an upstream fix that correctly
excludes discrete GPUs from a Unified-Memory-Architecture memory-reporting
path. The stale binary reported free memory from `/proc/meminfo` (system
RAM) instead of `cudaMemGetInfo` (real VRAM), so the sizing was reading the
wrong number entirely. Fixed by rebuilding the feeder's engine from the
host's pinned commit.

**Bug: llama.cpp's own `-fit` auto-placement conflicts with an explicit
`-ngl`.** Even after the fix above, EVERY layer count tried — 999 down to
11 of 65 — was rejected with the identical message:

```
common_fit_params: failed to fit params to free device memory:
  n_gpu_layers already set by user to N, abort
```

The flat number never mattered. Reading `common/fit.cpp` directly: for a
single device, if the params as given already satisfy the memory margin,
`-fit` returns early with no error at all — that's why solo-host loads
never hit this. For a **multi-device** load, if even one device's margin
isn't already satisfied as given, `-fit` proceeds toward an unconditional
check that throws on ANY explicit `-ngl`, independent of its value.
GreenBoost always sets `-ngl` explicitly — that's the whole point of its
own placement planner — so this fired every single time. Fixed with
`-fit off`: GreenBoost already owns this decision, llama.cpp shouldn't
re-veto it.

> **Lesson:** an identical error message across a wide sweep of otherwise-
> sensible values is a strong signal the *value* isn't the problem — read
> the check itself before tuning the number harder.

**Why it isn't used in production anyway.** With both bugs fixed, placement
correctly climbed to ~48/65 layers across the cluster (up from 20-28/65
solo) and real weight transfer to the feeder was confirmed (its VRAM
climbing as bytes streamed over the LAN). But load-plus-first-decode never
completed cleanly across several patient attempts — a pattern (a 502, then
minutes of 503s after weights had visibly finished loading) that looks like
genuine per-token network round-trip latency: this model's Gated Delta Net
layers on the feeder's side of the split need a cross-node round trip on
*every forward pass*, over a plain 1 Gbps LAN, not a low-latency
interconnect. Owner decision: stay host-only until someone measures RPC
*decode* speed specifically (not just load success) and it turns out
faster. The fixes stay landed regardless — they're correct behavior either
way, this is a policy choice about which placement to serve from, not a
retraction of the fixes.

---

## 4. MTP: still free speed, still provably identical output

This repo ships its own MTP draft head merged directly into the
`MTP`-prefixed quant files — no separate draft model, no extra VRAM.
GB-Synapse detects it from the GGUF's own tensors and enables llama.cpp's
speculative path:

```
--spec-type draft-mtp --spec-draft-n-max 3
```

Live-measured draft acceptance on this model: **64%** (41 of 64 drafted
tokens accepted in one real decode). Speculative decoding drafts cheaply,
the trunk verifies in one batched pass, and rejected drafts are discarded —
the output distribution is the trunk's, exactly, regardless of acceptance
rate. Acceptance rate only affects how much of the "free" speedup you
actually collect.

---

## 5. KV cache and context: clamped hard at first — because of a real bug, not the hardware

This model's native context is **262,144 tokens**. GB-Synapse clamps it down
to fit the live VRAM budget rather than OOMing mid-generation:

```
ctx=262144 would need 32.50 GB KV cache on top of 17.22 GB weights
  (10.01 GB budget) — clamping to ctx=2048
```

That **32.50 GB was wrong**, and it's worth walking through why, because the
correction changed the model from "barely usable" to "actually agentic."
This architecture is hybrid — `16 × (3 × Gated DeltaNet → 1 × Gated
Attention)` — only 1 layer in 4 holds a real, per-token-growing KV cache;
the other 3 (Gated DeltaNet, a linear-attention mechanism) hold a
**fixed-size recurrent state that does not grow with context length at
all**. `estimate_kv_gb()` didn't know this distinction and multiplied by
the full layer count uniformly (65, not the 17 that actually matter),
overestimating KV size by ~3.8x. Confirmed directly from the GGUF's own
`full_attention_interval=4` field, and cross-checked in llama.cpp's own
loader (`is_recr_impl` in `src/models/qwen35.cpp`) — the engine has always
known this internally; GreenBoost's own sizing math just didn't.

Fixed by reading `full_attention_interval` generically and deriving
`ModelEntry.n_kv_layers` from it (1 for a plain transformer — a safe no-op
for every other architecture). **Live result, same hardware, same quant,
same VRAM budget: served ctx went 2048 → 7680, a real 3.75x.** That crossed
a genuine threshold: GB-CLI's own baseline tool/system-prompt overhead
(~4608 tokens) now fits, so a real `gb -p "..."` request can actually think,
call a tool, and read project files — verified live, not assumed. It still
can't complete a full multi-tool-call audit in one turn (real file content
can exceed 7680 tokens on its own), so this isn't "solved," but it's a
different, much better problem than "can't start at all."

`q8_0` KV is used here as a **budget grade**, not a certification grade —
labelled as such in telemetry, same rule as always: a retrieval score is
meaningless without knowing which tier produced it.

---

## 6. Quality floor: measured, and honestly incomplete

`Q4_K_M` is nominally below fp8. The floor is enforced empirically, not
nominally:

- **`smoke_gate()`**, run 2026-07-27 against the real host-only serving
  path (28/65 layers on GPU, `q8_0` KV, MTP active): **PASS** — no
  repetition collapse (`max6gram=1`, `uniq=1.0`).
- **`niah_certify()`** — **not yet run.** The current `ctx=2048` clamp is
  too small for a meaningful long-context needle test; certifying a number
  at a context this small wouldn't reflect how the model is actually
  reachable at scale. Re-run once ctx headroom improves (more VRAM, a
  working feeder path, or a smaller quant), not before.

This model's own marketing (heavy fine-tune, "uncensored/heretic" merge)
makes measurement more important, not less — a model-card claim is a proxy
for quality, and when proxy and measurement disagree, the measurement wins.

---

## 7. Watching it: dataflux + MCP

Every decision above is real telemetry, queryable live:

```python
dataflux_events(kind="synapse_serve")   # why did it place things this way?
dataflux_events(kind="bench_result")    # the load/decode measurements from this doc
synapse_status()                        # what is actually running right now?
cluster_status()                        # is the feeder online, and is it worth using?
```

The RPC investigation above is a good example of why this matters: the
symptom (identical rejection at every layer count) only made sense once the
underlying `common_fit_params` source was read directly — no amount of
staring at dataflux numbers alone would have found it, but dataflux is what
confirmed the fix actually worked afterward (real weight transfer, no more
spurious rejections) rather than trusting a plausible-sounding theory.

---

## 8. The pipeline, as it actually runs today

```
gb  →  GB-CLI
      │  (OpenAI/Ollama API on :11435)
      ▼
   GB-Synapse proxy
      │
      ▼
   llama-server  (HOST ONLY — see §3 for why the RPC path stays unused)
      ├── -ngl 28                              as many layers as VRAM allows
      ├── --cache-type-k/v q8_0                budget grade (f16 doesn't fit at native ctx)
      ├── --ctx-size 2048                       clamped from 262144, see §5
      ├── --spec-type draft-mtp                 +free decode, ~64% acceptance, identical output
      └── --fit off                             GreenBoost owns placement, not llama.cpp's auto-fit
      │
      ▼
   GB-Dataflux  ──► MCP (synapse_serve, bench_result, smoke_gate, ...)
```

Nothing here is a heuristic someone liked the sound of, including the parts
that didn't work: the RPC split is real, tested, and its two blocking bugs
are genuinely fixed — it's excluded from production because the *measured*
behavior (a load that never cleanly finished decoding) outweighed the
*theoretical* case for it (more VRAM, more layers on GPU). That is the whole
job: read what the model is, read what the hardware is, measure rather than
assume, and say so plainly when the answer is "not yet."
