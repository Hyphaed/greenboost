"""
gb_llm.py — gb-quant for LLM inference (transformers / vLLM / ollama).

Same fit-first policy as gb_quant for diffusion: weights resident in T1 VRAM
at the highest precision that fits (bf16 > int8 > int4), T2 DDR as the only
overflow, T3 NVMe never for speed-critical weights.

Three runtimes, three levers:

1. transformers (this module) — full gb-quant treatment. Any HF causal LM is
   loaded on CPU and quantized-to-fit onto the GPU through gb_quant:

       import gb_llm
       model, tok = gb_llm.load_causal_lm("google/gemma-3-12b-it-qat-q4_0-unquantized")
       print(gb_llm.generate(model, tok, "Hola, com estàs?"))

2. vLLM — the gb-quant backend ships a native vLLM plugin (registered under
   the historical name "gemlite": `vllm.general_plugins` entry point
   gemlite.vllm.backend:register). Entry points need a real install, so in
   the vLLM venv run once:

       pip install -e ~/Dev/turboquantsolutions/gemlite  # dev checkout with
       # setup.py; same code greenboost carries in third_party/

   then serve with on-the-fly gb-quant weights:

       vllm serve <model> --quantization gemlite

   (KV-cache side: gb_attn / turboquant cover compression there.)

3. ollama — GGUF/llama.cpp runtime: no Python in-process hook is possible.
   The equivalent levers are (a) the GreenBoost shim, which already
   intercepts ollama's CUDA allocations for T1+T2 placement, and (b) picking
   the GGUF quant level at pull time so the model fits T1 (the manual
   analogue of plan_fit). Models like qwen3.6:latest are already 4-bit GGUF.
"""
from __future__ import annotations

import torch

import gb_quant

# Bootstrap all GreenBoost layers; no-op when not active.
try:
    import gb_init as _gb_init
except ImportError:
    _gb_init = None


def _auto_budget_gb() -> float:
    """Use gb_init unified budget (telemetry-first, torch fallback)."""
    if _gb_init is not None:
        return _gb_init.auto_budget_gb()
    try:
        free_b, _ = torch.cuda.mem_get_info()
        return free_b / 2**30 * 0.92
    except Exception:
        return 0.0


def load_causal_lm(model_id: str, budget_gb: "float | None" = None,
                   device: str = "cuda", dtype=torch.bfloat16,
                   cache_dir: "str | None" = None, trust_remote_code: bool = False,
                   verbose: bool = True, **hf_kwargs):
    """Load an HF causal LM quantized-to-fit through the gb-quant layer.

    Loads on CPU first (never OOMs the GPU on load), plans per-component
    precision against `budget_gb` (default: 92% of currently free VRAM), then
    realises the plan layer-by-layer onto `device`. Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pre-inference check: ECC errors and VRAM headroom via telemetry singleton.
    if _gb_init is not None:
        _gb_init.pre_inference_check()

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir,
                                        trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="cpu", low_cpu_mem_usage=True,
        cache_dir=cache_dir, trust_remote_code=trust_remote_code, **hf_kwargs,
    )
    if budget_gb is None:
        budget_gb = _auto_budget_gb()

    # NVTX range so the quantize+load span is visible in Nsight Systems.
    _gs = _gb_init.get_stream_sched() if _gb_init else None
    _nvtx_ctx = None
    if _gs is not None:
        try:
            import nvtx
            _nvtx_ctx = nvtx.annotate(
                message=f"gb:llm_load:{model_id.split('/')[-1]}",
                color="orange", domain="GreenBoost",
            )
            _nvtx_ctx.__enter__()
        except Exception:
            _nvtx_ctx = None

    gb_quant.quantize_to_fit(model, budget_gb=budget_gb, device=device,
                             dtype=dtype, verbose=verbose)
    # Move what the plan kept in bf16 (embeddings, norms, lm_head) to the GPU
    # on the transfer stream for non-blocking overlap.
    if _gs is not None:
        with _gs.on("transfer"):
            model.to(device, non_blocking=True)
        _gs.wait_for("transfer", on="gemm")
    else:
        model.to(device)
    model.eval()

    if _nvtx_ctx is not None:
        try:
            _nvtx_ctx.__exit__(None, None, None)
        except Exception:
            pass

    if verbose and _gb_init is not None:
        m = _gb_init.snapshot()
        if m is not None:
            print(
                f"[gb_llm] loaded — VRAM {m.fb_used_mb}/{m.fb_total_mb} MB "
                f"({m.fb_used_pct:.0f}%)  pwr={m.power_w:.0f}W  "
                f"temp={m.temp_c:.0f}°C",
                flush=True,
            )
    return model, tok


def generate(model, tok, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.7, **gen_kwargs) -> str:
    """Small convenience wrapper for smoke tests and scripts."""
    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt")
    except Exception:
        ids = tok(prompt, return_tensors="pt")
    # Tokenizers return either a tensor or a BatchEncoding dict.
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=temperature,
                             **gen_kwargs)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
