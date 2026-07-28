#!/usr/bin/env python3
"""gb_synapse_mcp.py — GB-Synapse MCP (server `greenboost-synapse`).

Serving control for GreenBoost's own model server (llama.cpp `--rpc`
tensor-split across the cluster + the Ollama/OpenAI proxy on the gb-synapse
port, default 11435, GB_SYNAPSE_PORT) PLUS the greenboost-cli bridge, so
scripts and LLMs use the whole greenboost-cli logic with ease:

  * `cli_run(subcommand, args)`   — allowlisted READ-ONLY headless gb
    subcommands (JSON output via greenboost_cli.cli_headless).
  * `cli_prompt(prompt, model)`   — one-shot `gb -p` prompt; the CLI's
    inference registry is gb-synapse-only (default :11435/v1) and
    auto-starts it, so this is inference-through-gb-synapse by construction.

Programmatic use from scripts (no MCP needed):
  * `from greenboost_cli.cli_headless import dispatch` — same subcommands, in
    process.
  * `gb -p "<prompt>"` / `gb <headless-subcommand> --llm` from any shell.

Safety: `synapse_serve` / `synapse_stop` are DRY-RUN unless `confirm=True` —
the gb-synapse port may be serving live consumers (ai-forge pipelines via
FORGE_OLLAMA_URL); changing what serves there is an owner-visible action,
never an implicit one.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("greenboost-synapse")

# Read-only headless gb subcommands (greenboost_cli/cli_headless.py dispatch).
# Anything not listed is rejected — mutations stay human/CLI actions.
_CLI_ALLOWLIST = ("rag-search", "rag-status", "crag-search", "crag-status",
                  "tokens", "skill-list", "skill-show", "plan-list",
                  "task-list", "convert", "compress")


def _gb_bin() -> str | None:
    return shutil.which("gb") or shutil.which("greenboost-cli")


@mcp.tool()
def synapse_status() -> dict:
    """GB-Synapse status: engine built (llama-server + rpc-server) + version,
    and whether the server and the gb-synapse Ollama/OpenAI proxy (default
    port 11435, GB_SYNAPSE_PORT) are running. Canonical owner (shared impl
    in gb_mcp_common.py — also mirrored on greenboost-orchestrator and
    greenboost-dataflux)."""
    import gb_mcp_common
    return gb_mcp_common.synapse_status()


@mcp.tool()
def synapse_models() -> list[dict]:
    """Models in the gb-synapse manifest (pulled from HF via `pull`, or
    imported from ollama via `index-ollama`) with name/engine/quant/format/
    size/source. `format` is detected from the model's own files (gguf/
    safetensors/diffusers/unknown) — independent of `engine`, which is the
    manifest's routing decision (may differ if the manifest is stale)."""
    try:
        import gb_synapse
        from gb_synapse_backends import detect_format
        out = []
        for m in gb_synapse.list_models():
            d = asdict(m)
            d["format"] = detect_format(m.path) if m.path else "unknown"
            out.append(d)
        return out
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
def synapse_ps() -> list[dict]:
    """Currently-served gb-synapse models (port, tensor-split)."""
    try:
        import gb_synapse
        return gb_synapse.ps()
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
def synapse_recommend(ctx: int = 65536, probe_feeders: bool = True) -> list[dict]:
    """Fit reports for every manifest model at the given context: does it fit
    VRAM / cluster, estimated (or measured) tok/s, and placement notes —
    measured history preferred over estimates."""
    try:
        import gb_synapse
        return [asdict(r) for r in gb_synapse.recommend(ctx=ctx,
                                                        probe_feeders=probe_feeders)]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
def synapse_doctor(probe_feeders: bool = True) -> dict:
    """GB-Synapse environment diagnosis: engine, CUDA, cluster rpc reachability,
    manifest health, plus `torch_engine_ready`/`torch_engine_env` (the
    synapse torch engine's venv, if installed)."""
    try:
        import gb_synapse
        return gb_synapse.doctor(probe_feeders=probe_feeders)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def synapse_serve(model: str, ctx: int = 65536, use_cluster: bool = True,
                  engine: str = "", confirm: bool = False,
                  cuda_graph: "bool | None" = None) -> dict:
    """Serve a model through gb-synapse (engine backend + the gb-synapse
    proxy, default port 11435, GB_SYNAPSE_PORT; tensor-split across the
    cluster when use_cluster and feeders are up for the llama.cpp backend).
    `engine` ("llama.cpp"/"torch"/"diffusers" — "vllm"/"transformers" still
    accepted as deprecated aliases for "torch") only affects the DRY-RUN
    preview's backend name — it does NOT override the manifest's own engine
    for an actual (confirm=True) serve; re-pull with --engine to change
    that durably.

    cuda_graph: synapse torch engine only (ignored by llama.cpp/diffusers).
    None (default) = GB_SYNAPSE_TORCH_CUDA_GRAPH env var (off by default,
    graph-capture warmup buffers can OOM small cards on top of the KV
    cache). True/False overrides per-call — e.g. worth trying once ctx is
    small enough to leave headroom (added 2026-07-28: this MCP tool had no
    way to reach this env-gated knob at all before, the exact kind of gap
    CLAUDE.md's MCP-tool-gap rule now says to close, not work around).

    DRY-RUN unless confirm=True: without it, returns the resolved model entry,
    which backend would serve it, and a warning about whatever is currently
    bound to the gb-synapse port — the owner (or an explicitly-authorized
    flow) decides to displace it."""
    try:
        import gb_synapse
        from gb_synapse_backends import select_backend
        entry = next((m for m in gb_synapse.list_models()
                      if m.name == model or model.lower() in m.name.lower()), None)
        if entry is None:
            return {"error": f"model '{model}' not in the manifest — pull it "
                             f"(gb_synapse pull) or index-ollama first",
                    "available": [m.name for m in gb_synapse.list_models()]}
        current = gb_synapse.ps()
        preview_engine = engine or entry.engine
        backend_name = select_backend(
            type(entry)(**{**asdict(entry), "engine": preview_engine})).name
        if not confirm:
            return {"dry_run": True, "would_serve": asdict(entry), "backend": backend_name,
                    "ctx": ctx, "use_cluster": use_cluster, "cuda_graph": cuda_graph,
                    "currently_serving": current,
                    "warning": f"the gb-synapse port ({gb_synapse.DEFAULT_PORT}) may "
                               "have live consumers (ai-forge FORGE_OLLAMA_URL, ollama "
                               "clients). Pass confirm=True to actually serve."}
        st = gb_synapse.serve(entry.name, ctx=ctx, use_cluster=use_cluster,
                              cuda_graph=cuda_graph)
        return {"serving": asdict(st)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def synapse_stop(model: str, confirm: bool = False) -> dict:
    """Stop a served gb-synapse model. DRY-RUN unless confirm=True (same
    gb-synapse-port protection as synapse_serve)."""
    try:
        import gb_synapse
        if not confirm:
            return {"dry_run": True, "would_stop": model,
                    "currently_serving": gb_synapse.ps(),
                    "warning": "pass confirm=True to actually stop"}
        return {"stopped": gb_synapse.stop(model)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def serve_and_repoint(model: str, port: int = 0,
                      forge_url_target: str | None = None,
                      confirm: bool = False) -> dict:
    """ACTUATE (double-gated): the one-step "prefer gb-synapse" action , serve
    `model` via gb-synapse AND repoint FORGE_OLLAMA_URL (in the shared
    inference.env) so ai-forge pipelines route through the gb-synapse proxy
    (cross-GPU --rpc split + gb-quant + dataflux tok/s) instead of raw ollama.
    DRY-RUN unless confirm=True AND GB_ORCH_ACTUATE=1. Replaces the old
    two-step dance (synapse_serve on one server + an out-of-band env edit)."""
    import gb_actuation
    return gb_actuation.serve_and_repoint(model, port=port,
                                          forge_url_target=forge_url_target,
                                          confirm=confirm)


@mcp.tool()
def cli_run(subcommand: str, args: list[str] = [], timeout: int = 120) -> dict:
    """Run an allowlisted READ-ONLY greenboost-cli headless subcommand and
    return its JSON. Allowlist: rag-search, rag-status, crag-search,
    crag-status, tokens, skill-list, skill-show, plan-list, task-list,
    convert, compress. Uses the installed `gb` (system wrapper or dev
    symlink); scripts can equally call
    `from greenboost_cli.cli_headless import dispatch` in-process."""
    if subcommand not in _CLI_ALLOWLIST:
        return {"error": f"subcommand '{subcommand}' not allowlisted "
                         f"(read-only bridge). Allowed: {list(_CLI_ALLOWLIST)}"}
    gb = _gb_bin()
    if not gb:
        return {"error": "gb / greenboost-cli not on PATH — run greenboost "
                         "Full Install (installs /usr/local/bin/gb) or the "
                         "cli's dev install"}
    try:
        r = subprocess.run([gb, subcommand, *[str(a) for a in args]],
                           capture_output=True, text=True, timeout=timeout)
        body = r.stdout.strip()
        try:
            return {"rc": r.returncode, "result": json.loads(body)}
        except Exception:
            return {"rc": r.returncode, "stdout": body[-4000:],
                    "stderr": r.stderr.strip()[-1000:]}
    except subprocess.TimeoutExpired:
        return {"error": f"gb {subcommand} timed out after {timeout}s"}


@mcp.tool()
def cli_prompt(prompt: str, model: str | None = None, timeout: int = 300) -> dict:
    """One-shot prompt through greenboost-cli (`gb -p`) — the CLI's registry is
    gb-synapse-only and auto-starts it, so the answer comes through the
    gb-synapse cluster path (cross-GPU split when serving with --rpc)."""
    gb = _gb_bin()
    if not gb:
        return {"error": "gb / greenboost-cli not on PATH"}
    cmd = [gb, "-p", prompt] + (["-m", model] if model else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"rc": r.returncode, "answer": r.stdout.strip()[-8000:],
                "stderr": r.stderr.strip()[-1000:] if r.returncode else ""}
    except subprocess.TimeoutExpired:
        return {"error": f"gb -p timed out after {timeout}s"}


if __name__ == "__main__":
    mcp.run()
