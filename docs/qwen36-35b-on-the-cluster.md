# Running Qwen3.6-35B (1M) on a two-GPU cluster, and why it works

**Audience:** an AI engineering student who knows what a transformer is, has run
Ollama, and wants to understand what GreenBoost actually *does* when you type
`gb` and a 20 GiB model answers on a 12 GB graphics card.

This is a walkthrough of one real configuration, end to end:

| | |
|---|---|
| Model | `satgeze/qwen36-35b-uncensored-1m` (HF GGUF, `Q4_K_M`, **20.2 GiB**) |
| Architecture | `qwen35moe` , 35B total, ~3B active, **256 experts, 8 fire per token** |
| Context | 1,048,576 tokens (YaRN-baked), MTP layer grafted in, vision tower attached |
| Host | RTX 5070, **12 GB** VRAM · 64 GB DDR4 |
| Feeder | RTX 5070 Laptop, **8 GB** VRAM · 32 GB DDR5, over LAN |
| Served by | GB-Synapse → `llama-server` + `--rpc`, on `:11434` |

The whole problem in one line: **20.2 GiB of weights, ~17 GB of usable VRAM.**
Everything below is about closing that 3 GB gap without paying for it in speed
or quality.

---

## 1. Why the model file matters (and why the Ollama blob failed)

The same model exists as an Ollama blob and as a HuggingFace GGUF. They are not
interchangeable:

```
llama_model_load: error loading model hyperparameters:
  key qwen35moe.rope.dimension_sections has wrong array length; expected 4, got 3
```

Ollama's blob carries **3** rope dimension sections; upstream llama.cpp expects
**4**. Ollama can run it because Ollama ships its *own* engine. GB-Synapse serves
through llama.cpp, so it needs the artifact llama.cpp was built for , the HF
GGUF. Same weights, different metadata contract.

The same asymmetry bites vision: **Ollama's blobs carry no separate projector
layer** (it embeds vision in its own engine), so an Ollama-sourced VLM served
through llama.cpp is *text-only*. It will not error on an image , it will
confidently describe one it never received. Vision models must come from HF as
`model.gguf` **+** `mmproj.gguf`.

> **Lesson:** "the model" is not just weights. Metadata and companion files are
> part of the contract between a checkpoint and an engine.

---

## 2. Two GPUs, one device: the RPC split

GB-Cluster starts `rpc-server` on the feeder; llama.cpp then treats that remote
GPU as another backend device and splits the model's layers across both:

```
llama-server ... --rpc 192.168.50.246:50052 --tensor-split 8513,6034
```

The split ratio is computed from **real free VRAM per device**, minus each
device's compute/graph workspace. That subtraction is not a detail: sizing the
split from raw free VRAM handed a 7.5 GB feeder an 8.4 GB share, and on a GPU an
overshoot is not a spill, it is a hard failure:

```
alloc_tensor_range: failed to allocate RPC0[...] buffer of size 8441235200
```

Only **activations** cross the network each step , a few MB , not weights. That
is why a gigabit LAN is enough: you are shipping the thin part of the
computation, not the fat part.

---

## 3. The key insight: move the *experts*, not the *layers*

We are 3 GB short. The naive fix is to leave some layers off the GPU
(`-ngl 23` of 41). **That is the wrong 3 GB to move.** A layer you drop takes
its *attention* to the CPU, and attention is read for **every token**.

Look at what this model is actually made of (measured from the GGUF, not
assumed):

```
total 20.2 GiB | expert 18.6 GiB (92%) | non-expert 1.6 GiB
41 layers → 0.45 GiB of expert weight per layer
```

Two facts collide productively:

1. Experts are **92% of the bytes**.
2. Only **8 of 256** experts fire on any given token.

So experts are simultaneously the biggest thing in the model and the least
frequently touched. They are the cheapest bytes to evict from VRAM. llama.cpp
exposes exactly this:

```
-ngl 999 --n-cpu-moe 19
```

Read that as: *every one of the 41 layers stays on the GPU , attention, KV,
norms, routing , and only the **expert tensors** of the first 19 layers live in
DDR.* We keep the hot path (touched every token) in VRAM and exile the cold,
sparse, enormous part.

This is the same principle as GreenBoost's tiering rule , **access frequency,
not size, decides what occupies VRAM** , applied inside a single model.

---

## 4. MTP: free speed, provably identical output

Qwen3.6 ships a **multi-token-prediction** layer, and this build has it grafted
back at the tensor level. GB-Synapse detects it by inspecting the GGUF's tensors
(not its filename) and enables llama.cpp's speculative decoding:

```
--spec-type draft-mtp --spec-draft-n-max 3
```

Speculative decoding drafts several tokens cheaply, then the **trunk verifies
them in one batched pass**. Rejected drafts are discarded. This is why it is
*not* a quality tradeoff: the output distribution is the trunk's, exactly. And
unlike a separate draft model, the MTP head is part of this checkpoint , it
costs no extra VRAM. Upstream measures **+34% decode** on this model.

> **Lesson:** speculative decoding buys latency with *arithmetic*, not with
> accuracy. Always check whether a "speedup" changes the output distribution.
> This one provably does not.

---

## 5. KV cache: a quality tier, not a memory knob

The KV cache is re-read on every decode step, so its size drives both memory and
bandwidth. You can store it at `f16` or quantize it to `q8_0`.

GB-Synapse treats this as a **quality decision**:

- `f16` , **certification grade**. Chosen whenever it fits the budget.
- `q8_0` , **budget grade**. Used only to make a context length reachable at
  all, and *labelled as such* in telemetry.

That labelling matters because a retrieval score is meaningless without it. The
aviary NIAH ladder certifies at f16 and reports quantized-KV runs separately;
a "10/10 at 1M" that quietly used a compressed cache is a different claim.

Context is then clamped to what the budget can actually hold. A 1M-token f16 KV
for this model would need ~41 GB , more than the whole cluster , so the served
context is reduced until it fits, and gb-synapse says so out loud rather than
OOMing mid-generation.

---

## 6. Quality floor: what "at least fp8" really means here

GreenBoost's standing rule is **never drop below fp8-equivalent quality**. This
model is served at `Q4_K_M`, which is *nominally* below fp8. That deserves an
honest answer rather than a hand-wave:

- There is no fp8/Q8 build of this checkpoint that fits: Q8 would be ~35 GB
  against ~17 GB of cluster VRAM. Forcing it would spill massively and destroy
  the speed we are trying to buy.
- So the floor is enforced **empirically instead of nominally**:
  - **`niah_certify()`** , the shipped build is certified **70/70 needles to 1M**.
    Retrieval is measured, not assumed.
  - **`smoke_gate()`** , catches repetition-collapse, the failure signature of a
    quant pushed too low. A quant that passes this and the needle ladder is
    demonstrably intact, whatever its nominal bit-width.
  - **f16 KV** wherever it fits.

> **Lesson:** bit-width is a proxy for quality, not a measurement of it. When the
> proxy and the measurement disagree, trust the measurement , and publish the
> failures too.

---

## 7. Why we refuse CPU spillover (and what we still owe)

Two things can "overflow" when a model exceeds VRAM, and they are not remotely
equivalent:

| | What moves | What it costs |
|---|---|---|
| **CPU offload** (`-ngl` low, `--cpu-moe`) | the **computation** | the CPU has ~30× less compute throughput than the GPU. You now run matmuls on the wrong processor. |
| **GreenBoost tiering** | only the **memory** | the GPU reads the weights from pinned DDR over PCIe (~25 GB/s on Gen4 x16) and does the math itself. |

GreenBoost's design rule is that **compute never leaves the GPU.** The kernel
module pins DDR pages and hands them to CUDA (`cuImportExternalMemory` on a
DMA-BUF, or `cuMemHostRegister` with `DEVICEMAP`), so the GPU's own copy engines
read tensors straight out of system RAM. The step that is *skipped* is the one
CPU offload cannot avoid: there is **no staging copy into a host buffer and no
CPU-side matmul** , the pages are already in the GPU's address space.

**Where we honestly stand today:** the `--n-cpu-moe` path above is llama.cpp's
own offload, which means those expert FFNs are currently computed **on the CPU**.
That is a compromise, not the design target. The reason is a live bug: preloading
the GreenBoost shim into `llama-server` aborts ggml's CUDA backend
(`invalid device function` in `ggml_cuda_kernel_can_use_pdl`), so per-node T2 is
unavailable to llama.cpp right now. The target , experts resident in **T2 pinned
DDR, read by the GPU, computed by the GPU** , removes the CPU from the data path
entirely. Closing that bug converts the last CPU-side compute in this pipeline
into a PCIe read.

Saying otherwise would be exactly the kind of claim this document exists to
discourage.

---

## 8. Watching it: dataflux + MCP

Every decision above is emitted as telemetry, so none of it is folklore:

| Event | What it records |
|---|---|
| `synapse_serve` | model, arch, quant, ctx, KV tier, tensor split, feeders, MTP/vision, load time, and **failures with the engine's own error** |
| `tensor_split` | the ratio and the free VRAM it came from |
| `tok_s_measured` | client-observed throughput, which closes the loop on every decision above |
| `niah_cert` / `smoke_gate` / `yarn_bake` | the quality evidence , misses published alongside passes |

Query it live from any MCP-capable assistant:

```python
dataflux_events(kind="synapse_serve")   # why did it place things this way?
dataflux_tok_s()                        # what did that buy?
synapse_status()                        # what is actually running right now?
cluster_status()                        # is the feeder really carrying weight?
```

That last one is not rhetorical. During this very bring-up, `synapse_status`
reported `server_running: false` while the CLI was cheerfully printing
`✓ gb-synapse started (pid 81632)` , which is how a zombie-process bug got
caught. **The flux is the evidence; the log line is just a story.**

---

## 9. The whole pipeline, in order

```
gb  →  GB-CLI
      │  (OpenAI/Ollama API on :11434)
      ▼
   GB-Synapse proxy ── translates Ollama `images` → OpenAI image_url
      │
      ▼
   llama-server
      ├── --rpc feeder:50052 ──────────────► feeder GPU (RTX 5070 8 GB)
      ├── --tensor-split 8513,6034            (activations cross the LAN, not weights)
      ├── -ngl 999                            every layer's attention + KV on GPU
      ├── --n-cpu-moe 19                      experts of 19 layers → DDR
      ├── --spec-type draft-mtp               +34% decode, identical output
      ├── --cache-type-k/v                    f16 if it fits, else q8_0 (labelled)
      └── --mmproj                            vision tower
      │
      ▼
   GB-Dataflux  ──► MCP (synapse_serve, tok_s_measured, niah_cert, ...)
```

Nothing here is a heuristic someone liked the sound of. Each flag is a
consequence of a measured property of the model (92% experts, 8/256 active, an
MTP head, a 4-section rope key) meeting a measured property of the machine
(12 GB + 8 GB of VRAM, a LAN, PCIe Gen4). That is the whole job: **read what the
model is, read what the hardware is, and place every byte where its access
frequency says it belongs.**
