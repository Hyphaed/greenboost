"""
/gb-quant — GreenBoost weight quantization for LLMs.

Usage:
  /gb-quant <model> [bits]              # in-process: load BF16 + gb-quant + test gen
  /gb-quant <model> [bits] --serve      # launch vLLM with --quantization gemlite
  /gb-quant --list                      # list GB_HOME/quant_models/

<model>: Ollama tag (e.g. qwen3.6:latest) or HuggingFace ID
[bits]:  fp8 (default), int8, 8, int4, 4, tq3, tq2, auto
"""
from __future__ import annotations

import sys
import threading

from greenboost_cli.terminal.theme import (
    emit_ok, emit_err, emit_warn, emit_info,
    console, VIOLET, GRAY, LIME, AMBER,
)
from greenboost_cli.environment.settings import GB_HOME
from greenboost_cli.terminal.commands import register_command

# Path to the GreenBoost source tree (gb_quant.py / gb_llm.py live here)
from greenboost_cli.gb_paths import gb_py_root, gb_root_hint

_GB_SRC = gb_py_root()

_VALID_BITS = {"fp8", "e4m3", "int8", "8", "int4", "4", "tq3", "tq2", "auto", "bf16"}

# Ollama tag → HuggingFace model ID
_QUANT_HF_MAP = {
    "qwen3.6":       "Qwen/Qwen3-3.6B-Instruct",
    "qwen3.6-27b":   "Brian6145/Qwen3.6-27B-Claude-Opus-Sonnet-Distilled-NVFP4-MTP",
    "qwen3":         "Qwen/Qwen3-8B",
    "qwen2.5-coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5":       "Qwen/Qwen2.5-7B-Instruct",
    "llama3.3":      "meta-llama/Llama-3.3-70B-Instruct",
    "llama3.2":      "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1":      "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-r1":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "phi4":          "microsoft/phi-4",
    "phi3":          "microsoft/Phi-3-mini-4k-instruct",
    "mistral":       "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral":       "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gemma3":        "google/gemma-3-4b-it",
    "gemma2":        "google/gemma-2-9b-it",
}


def _resolve_hf_id(name: str) -> str:
    """Map an Ollama tag or short name to a HuggingFace model ID."""
    base = name.split(":")[0].lower()
    if base in _QUANT_HF_MAP:
        return _QUANT_HF_MAP[base]
    for key, hf_id in _QUANT_HF_MAP.items():
        if base.startswith(key):
            return hf_id
    return name   # assume raw HF ID (e.g. "Qwen/Qwen3-8B")


def _normalize_bits(bits: str):
    """Normalise user bit-width string to a gb_quant-accepted value (int or str)."""
    b = bits.strip().lower()
    if b in ("int4", "4"):
        return 4
    if b in ("int8", "8"):
        return 8
    if b in ("fp8", "e4m3"):
        return "fp8"
    if b in ("tq3", "tq2", "bf16"):
        return b
    return "fp8"   # safe default for unknown/auto on Blackwell


def _ensure_gb_path() -> None:
    if str(_GB_SRC) not in sys.path:
        sys.path.insert(0, str(_GB_SRC))


def _list_quant_models() -> None:
    quant_dir = GB_HOME / "quant_models"
    if not quant_dir.exists() or not any(quant_dir.iterdir()):
        emit_info("No saved quantized models in ~/.greenboost_cli/quant_models/")
        return
    console.print(f"\n[bold {VIOLET}]Saved quantized models[/]")
    console.print(f"[{GRAY}]{'─' * 50}[/]")
    for d in sorted(quant_dir.iterdir()):
        if d.is_dir():
            size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) // (1024 * 1024)
            console.print(f"  [{LIME}]{d.name:<42}[/]  [{GRAY}]{size_mb} MB[/]")
    console.print()


def _run_inprocess(hf_id: str, bits, _session, _settings: dict) -> None:
    """Load BF16 HF weights on CPU, apply gb-quant at the requested precision, run a test."""
    _ensure_gb_path()

    try:
        import gb_quant
    except ImportError as e:
        emit_err(f"Cannot import gb_quant from {_GB_SRC}: {e}")
        emit_info(f"Fix: {gb_root_hint()}.")
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        emit_err("Missing deps: pip install transformers accelerate torch")
        return

    result: dict = {}
    done_event = threading.Event()

    def _load_and_quant():
        try:
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            if bits == "auto":
                free_b, _ = torch.cuda.mem_get_info()
                budget_gb = free_b / 2**30 * 0.92
                gb_quant.quantize_to_fit(model, budget_gb=budget_gb, verbose=True)
            else:
                gb_quant.quantize_module(model, bits=bits)
            model.to("cuda")
            model.eval()
            result["model"] = model
            result["tok"]   = tok
        except Exception as exc:
            result["error"] = exc
        finally:
            done_event.set()

    t = threading.Thread(target=_load_and_quant, daemon=True)
    t.start()

    from rich.progress import Progress, SpinnerColumn, TextColumn
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        console=console, transient=True,
    ) as progress:
        progress.add_task(
            f"[{VIOLET}]Loading[/] [{GRAY}]{hf_id}[/] [{LIME}]{bits}[/] …", total=None
        )
        while not done_event.wait(timeout=0.2):
            pass

    if "error" in result:
        emit_err(f"Load failed: {result['error']}")
        return

    model = result["model"]
    tok   = result["tok"]

    # VRAM stats
    used_mb, total_mb = (v // (1024 * 1024) for v in (
        torch.cuda.memory_allocated(), torch.cuda.get_device_properties(0).total_memory
    ))
    emit_ok(f"Loaded  {hf_id}  bits={bits}  VRAM {used_mb}/{total_mb} MB")

    # Quick smoke test
    try:
        from greenboost_cli.slash_commands.backend_cmds import _ensure_gb_path as _backend_path  # noqa
    except Exception:
        pass
    _ensure_gb_path()
    try:
        import gb_llm
        reply = gb_llm.generate(model, tok, "Hello! What can you do?", max_new_tokens=64)
        console.print(f"\n[{GRAY}]Test generation:[/]")
        console.print(f"[{LIME}]{reply.strip()}[/]\n")
    except Exception as e:
        emit_warn(f"Test generation error: {e}")


def cmd_gb_quant(args: str, session, settings: dict) -> bool:
    """Quantize an LLM with GreenBoost gb-quant weight quantization.

    /gb-quant <model> [bits]  — in-process load + quantize + test
    /gb-quant --list           — list saved quantized models

    Note: the old `--serve` mode (vLLM + gemlite plugin) was removed when
    greenboost-cli standardized on gb-synapse as its only inference backend
    — gemlite-quantized safetensors have no gb-synapse (GGUF/llama.cpp)
    equivalent. To serve a quantized model, pull a GGUF quant instead:
    greenboost pull <org/repo>[:quant]  then  /llamaserve <name>.
    """
    parts = args.strip().split()

    if not parts or parts[0] in ("--list", "-l", "list"):
        _list_quant_models()
        return True

    if "--serve" in [p.lower() for p in parts]:
        emit_err("--serve was removed with vLLM. Use gb-synapse instead:")
        emit_info("  greenboost pull <org/repo>[:quant]  then  /llamaserve <name>")
        return True

    model_name = parts[0]
    bits_str   = "fp8"   # FP8 is the correct default on Blackwell (NVFP4 Triton blocked)

    i = 1
    while i < len(parts):
        p = parts[i].lower()
        if p in _VALID_BITS:
            bits_str = p
        i += 1

    hf_id = _resolve_hf_id(model_name)
    bits  = _normalize_bits(bits_str)

    console.print(
        f"\n  [{GRAY}]model:[/]  [{VIOLET}]{model_name}[/]  [{GRAY}]→[/]  [{LIME}]{hf_id}[/]\n"
        f"  [{GRAY}]bits: [/]  [{LIME}]{bits}[/]  [{GRAY}]  mode:[/]  [{VIOLET}]in-process[/]\n"
    )

    _run_inprocess(hf_id, bits, session, settings)

    return True


register_command("gb-quant", cmd_gb_quant, "Quantize an LLM with gb-quant (FP8/INT4/TQ3)")
