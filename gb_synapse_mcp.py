#!/usr/bin/env python3
"""gb_synapse_mcp.py — GB-Synapse MCP (server `greenboost-synapse`).

Serving control for GreenBoost's own model server (llama.cpp `--rpc`
tensor-split across the cluster + the Ollama/OpenAI proxy on the gb-synapse
port, default 11369, GB_SYNAPSE_PORT) PLUS the greenboost-cli bridge, so
scripts and LLMs use the whole greenboost-cli logic with ease:

  * `cli_run(subcommand, args)`   — allowlisted READ-ONLY headless gb
    subcommands (JSON output via greenboost_cli.cli_headless).
  * `cli_prompt(prompt, model)`   — one-shot `gb -p` prompt; the CLI's
    inference registry is gb-synapse-only (default :11369/v1) and
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
from mcp.types import ToolAnnotations  # noqa: E402

mcp = FastMCP("greenboost-synapse")

# Per-tool MCP read-only declaration (AE-6). Judged one tool at a time, never
# swept across the file: this server also carries tools that actuate, serve,
# stop, dispatch or write policy, and mislabelling one of those as read-only
# would let a client run it concurrently with anything else. `readOnlyHint` is
# the protocol's own answer to "may this overlap"; a tool that does not carry
# it stays serial, which is the safe default and the previous behaviour.
_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

# Read-only headless gb subcommands (greenboost_cli/cli_headless.py dispatch).
# Anything not listed is rejected — mutations stay human/CLI actions.
_CLI_ALLOWLIST = ("rag-search", "rag-status", "crag-search", "crag-status",
                  "tokens", "skill-list", "skill-show", "plan-list",
                  "task-list", "convert", "compress")


def _gb_bin() -> str | None:
    return shutil.which("gb") or shutil.which("greenboost-cli")


@mcp.tool(annotations=_READ_ONLY)
def synapse_status() -> dict:
    """GB-Synapse status: engine built (llama-server + rpc-server) + version,
    and whether the server and the gb-synapse Ollama/OpenAI proxy (default
    port 11369, GB_SYNAPSE_PORT) are running. Canonical owner (shared impl
    in gb_mcp_common.py — also mirrored on greenboost-orchestrator and
    greenboost-dataflux)."""
    import gb_mcp_common
    return gb_mcp_common.synapse_status()


@mcp.tool(annotations=_READ_ONLY)
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


@mcp.tool(annotations=_READ_ONLY)
def synapse_ps() -> list[dict]:
    """Currently-served gb-synapse models (port, tensor-split)."""
    try:
        import gb_synapse
        return gb_synapse.ps()
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool(annotations=_READ_ONLY)
def synapse_recommend(ctx: int = 65536, probe_feeders: bool = True,
                      preview_recipe: bool = False) -> list[dict]:
    """Fit reports for every manifest model at the given context: does it fit
    VRAM / cluster, estimated (or measured) tok/s, and placement notes —
    measured history preferred over estimates.

    preview_recipe=True (NemoClaw audit, Phase 5e MCP surface) adds a
    `recipe` key to each report: null when no validated serving recipe
    matches that model, else the recipe's own ctx/nGpuLayers/kvCache/
    mtpDraftN/tierIntent — "would this model's serve() actually use a
    pinned recipe instead of the heuristics" without serving anything."""
    try:
        import gb_synapse
        reports = [asdict(r) for r in gb_synapse.recommend(ctx=ctx,
                                                           probe_feeders=probe_feeders)]
        if preview_recipe:
            for report in reports:
                recipe = gb_synapse.load_recipe(report.get("model", ""))
                report["recipe"] = {
                    "ctx": recipe["ctx"], "nGpuLayers": recipe["nGpuLayers"],
                    "kvCache": recipe["kvCache"], "mtpDraftN": recipe.get("mtpDraftN"),
                    "tierIntent": recipe["tierIntent"],
                } if recipe else None
        return reports
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool(annotations=_READ_ONLY)
def synapse_doctor(probe_feeders: bool = True) -> dict:
    """GB-Synapse environment diagnosis: engine, CUDA, cluster rpc reachability,
    manifest health, plus `torch_engine_ready`/`torch_engine_env` (the
    synapse torch engine's venv, if installed), and a `recipes` block
    (NemoClaw audit, Phase 5e MCP surface) listing every serving recipe
    found under serving/recipes/ with its target model name and whether it
    currently validates (schema + digest) — "which models have a pinned
    recipe" without grepping the directory by hand."""
    try:
        import gb_synapse
        result = gb_synapse.doctor(probe_feeders=probe_feeders)
        result["recipes"] = _recipe_coverage_report()
        return result
    except Exception as e:
        return {"error": str(e)}


def _recipe_coverage_report() -> list[dict]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "serving"))
        import check_recipes as cr
    except ImportError:
        return []
    schema = cr._load_schema()
    report = []
    for path in cr.find_recipe_files():
        model_name = ""
        try:
            with open(path, encoding="utf-8") as f:
                import yaml
                recipe = yaml.safe_load(f)
            model_name = (recipe or {}).get("model", {}).get("name", "")
            cr.check_recipe(path, schema)
            report.append({"path": str(path), "model": model_name, "valid": True})
        except Exception as e:
            report.append({"path": str(path), "model": model_name, "valid": False, "error": str(e)})
    return report


@mcp.tool()
def synapse_serve(model: str, ctx: int = 65536, use_cluster: bool = True,
                  engine: str = "", confirm: bool = False,
                  cuda_graph: "bool | None" = None,
                  cache_ram: "int | None" = None,
                  mtp_draft_n: "int | None" = None,
                  spec_draft_p_min: "float | None" = None,
                  slot_prompt_similarity: "float | None" = None,
                  kv_type: "str | None" = None,
                  recipe: "str | None" = None,
                  use_preset: bool = False) -> dict:
    """Serve a model through gb-synapse (engine backend + the gb-synapse
    proxy, default port 11369, GB_SYNAPSE_PORT; tensor-split across the
    cluster when use_cluster and feeders are up for the llama.cpp backend).
    `engine` ("llama.cpp"/"torch"/"diffusers"/"video" — "vllm"/"transformers"
    still accepted as deprecated aliases for "torch") only affects the DRY-RUN
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

    cache_ram: llama.cpp backend only (ignored by torch/diffusers) — MiB
    size of the --cache-ram host-memory prompt cache. None (default) derives
    it from live free host RAM (see LlamaCppBackend.serve()); pass an
    explicit value to override for one serve call.

    mtp_draft_n: llama.cpp backend only, and only for models carrying a
    multi-token-prediction head (ignored otherwise) — --spec-draft-n-max for
    the MTP speculative-decode path. None (default) = GB_SYNAPSE_MTP_DRAFT_N
    env var (default 4, the 2026-08-05 sweep's winner). On a bandwidth-bound
    dense decode a forward pass
    costs the same wall time no matter how many tokens speculative decoding
    accepts from it, so raising this is quality-neutral throughput, not a
    tradeoff — same output distribution as unquantized greedy/sampled decode
    either way (added 2026-08-05: this MCP tool had no way to reach this
    env-gated knob at all before, same class of gap the cuda_graph param
    above closed on 2026-07-28, per CLAUDE.md's MCP-tool-gap rule).

    spec_draft_p_min: llama.cpp backend only, same MTP-head gate as
    mtp_draft_n — --spec-draft-p-min, the minimum draft-head top-token
    probability before a draft bails early instead of running to full
    mtp_draft_n depth regardless of confidence. None (default) =
    GB_SYNAPSE_MTP_P_MIN env var (default 0.3, the 2026-08-05 sweep's
    winner). Still
    exact verification either way — only affects how much compute a
    low-confidence draft burns before being discarded.

    slot_prompt_similarity: llama.cpp backend only — --slot-prompt-similarity,
    how much a new request's prompt must overlap an idle server slot's
    cached prompt before llama-server reuses that slot instead of LRU.
    None (default) = GB_SYNAPSE_SLOT_PROMPT_SIMILARITY (default 0.5, raised
    from this engine's own compiled default of 0.10 — live-tested
    2026-08-05: at 0.10, concurrent conversations sharing a system prompt
    all converged on matching only that shared prefix, never each
    conversation's own accumulated turns). This is what lets a multi-turn
    GB-CLI session's later turns skip re-prefilling their own growing
    history (measured: 93% cached tokens on a natural turn 2) instead of
    silently losing that reuse to slot round-robin every single turn.

    kv_type: llama.cpp backend only — --cache-type-k/-v (e.g. "f16"/"q8_0"/
    "q4_0"). None (default) leaves KV precision to the live budget-driven
    choice (_pick_kv_type()) or a recipe's kvCache field. An explicit value
    here wins over both — added 2026-08-10: before this, the only way to
    force a specific KV type through this tool was a raw `extra_args`
    flag, which llama.cpp's own arg parser silently ignored (it keeps the
    FIRST occurrence of a duplicate flag, and LlamaCppBackend.serve() always
    emits its own --cache-type-k/-v before appending extra_args) — exactly
    the MCP-Tool-Gaps rule's "fix the tool, don't work around it" case.

    recipe: an explicit serving recipe filename (looked up under
    serving/recipes/) or absolute path (NemoClaw audit, Phase 5e MCP
    surface) — schema+digest validated on load (serving/check_recipes.py);
    an invalid one is a hard `{"error": ...}`, never a silent fallback to
    heuristics (unlike gb_synapse.serve()'s own model-name auto-match via
    load_recipe(), which DOES fall back quietly since it runs on every
    serve whether or not the caller thought about recipes at all — an
    explicit `recipe=` here means the caller specifically asked for one).
    When given, its ctx/mtpDraftN win over this call's own `ctx`/
    `mtp_draft_n` parameters outright — pass a recipe OR pass explicit
    ctx/mtp_draft_n for this call, not both if they might disagree.

    use_preset: False (default — nothing changes) means "decide
    use_cluster from the `use_cluster` argument, as always." True (NemoClaw
    audit, Phase 7 follow-up — the resolver now CAN drive live dispatch,
    on request) hands that decision to serving/resolver.py's live-facts
    resolution instead: the winning preset's recipe's `tierIntent` decides
    use_cluster outright (rpcSplit -> True, t1Only/t2Spill -> False),
    ignoring whatever `use_cluster` was passed. The dry-run preview shows
    which preset WOULD be selected (and why, via the resolver's per-preset
    evaluations) without committing to anything; `confirm=True` propagates
    the same resolution to the real gb_synapse.serve() call, which raises
    (not falls back silently) on an ambiguous-selection tie or no eligible
    preset — an explicit ask for preset-driven dispatch that can't resolve
    is a real error, not a reason to guess.

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

        recipe_used = None
        if recipe is not None:
            recipe_path = Path(recipe)
            if not recipe_path.is_absolute():
                recipe_path = Path(__file__).resolve().parent / "serving" / "recipes" / recipe
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent / "serving"))
                import check_recipes as cr
                import yaml
                with open(recipe_path, encoding="utf-8") as f:
                    recipe_used = yaml.safe_load(f)
                cr.check_recipe(recipe_path, cr._load_schema())
            except Exception as e:
                return {"error": f"recipe {recipe!r} failed to load/validate: {e}"}
            ctx = int(recipe_used["ctx"])
            if "mtpDraftN" in recipe_used:
                mtp_draft_n = int(recipe_used["mtpDraftN"])

        current = gb_synapse.ps()
        preview_engine = engine or entry.engine
        backend_name = select_backend(
            type(entry)(**{**asdict(entry), "engine": preview_engine})).name

        preset_preview = None
        if use_preset:
            preset_preview = gb_synapse.resolve_serving_preset()

        if not confirm:
            return {"dry_run": True, "would_serve": asdict(entry), "backend": backend_name,
                    "ctx": ctx, "use_cluster": use_cluster, "cuda_graph": cuda_graph,
                    "cache_ram": cache_ram, "mtp_draft_n": mtp_draft_n,
                    "spec_draft_p_min": spec_draft_p_min,
                    "slot_prompt_similarity": slot_prompt_similarity, "kv_type": kv_type,
                    "currently_serving": current,
                    "recipe_applied": recipe_used, "preset_preview": preset_preview,
                    "warning": f"the gb-synapse port ({gb_synapse.DEFAULT_PORT}) may "
                               "have live consumers (ai-forge FORGE_OLLAMA_URL, ollama "
                               "clients). Pass confirm=True to actually serve."}
        st = gb_synapse.serve(entry.name, ctx=ctx, use_cluster=use_cluster,
                              cuda_graph=cuda_graph, cache_ram=cache_ram,
                              mtp_draft_n=mtp_draft_n,
                              spec_draft_p_min=spec_draft_p_min,
                              slot_prompt_similarity=slot_prompt_similarity,
                              kv_type=kv_type,
                              use_preset=use_preset)
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
def synapse_pause(model: str, force: bool = False, confirm: bool = False) -> dict:
    """ACTUATE: pause a serve session , save its KV state to disk, stop the
    engine, and return the VRAM. DRY-RUN unless confirm=True (same protection
    as synapse_stop: this takes a live :11369 endpoint down).

    The point is reclaiming a card that is allocated but not working. Measured
    on this box 2026-08-18: 10.4 GiB of 12 GB held at 0% GPU utilization with
    no way to get it back without losing the conversation. Slot save/restore
    makes it recoverable , 740 ms to save 54,377 tokens, 537 ms to restore.

    Refuses while a request is generating unless force=True, because llama.cpp
    cannot checkpoint a half-finished generation and that output would be lost.
    Check `idle_serve_holding_vram` (semantic_segments) to decide whether a
    pause is worth it at all."""
    try:
        import gb_synapse
        if not confirm:
            return {"dry_run": True, "would_pause": model,
                    "currently_serving": gb_synapse.ps(),
                    "warning": "pass confirm=True to actually pause"}
        return gb_synapse.pause(model, force=force)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def synapse_resume(model: str, confirm: bool = False) -> dict:
    """ACTUATE: resume a paused serve session , re-serve it with the same
    parameters and restore its saved KV state. DRY-RUN unless confirm=True.

    Reports reload_s (weight loading, the slow part) separately from
    restore_s (KV restore, sub-second) so the two are never conflated. A failed
    KV restore is reported but does NOT fail the resume: the session comes back
    with a cold cache rather than not at all."""
    try:
        import gb_synapse
        if not confirm:
            return {"dry_run": True, "would_resume": model,
                    "paused_sessions": gb_synapse.paused(),
                    "warning": "pass confirm=True to actually resume"}
        return gb_synapse.resume(model)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(annotations=_READ_ONLY)
def synapse_paused() -> dict:
    """Paused serve sessions waiting to be resumed , model, saved token count,
    disk held, and how long each has been paused. Read-only and cheap."""
    try:
        import gb_synapse
        return {"paused": gb_synapse.paused()}
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


@mcp.tool()
def quality_gate(model: str, gate: str = "smoke", tokens: int = 65536,
                 needles: int = 10, kv_type: str = "unknown",
                 seed: int = 1337) -> dict:
    """Run gb_aviary's quality gates against the LIVE-serving `model` — the
    empirical evidence CLAUDE.md's "never below fp8 without gate evidence"
    rule requires before accepting a below-fp8 quant. Before this tool
    (added 2026-08-05), `gb_aviary.smoke_gate()`/`niah_certify()` had zero
    invocation surface anywhere in the repo — no CLI, no MCP tool, the only
    caller (`gb_actuation.py`) just emitted an advisory STRING naming them
    without ever calling either. A policy that can't be exercised isn't
    enforced, just aspirational — the exact class of gap CLAUDE.md's
    MCP-Tool-Gaps rule says to close, not work around.

    gate="smoke" (default, cheap, ~seconds): `smoke_gate()` — one fixed
    prompt, checks for repetition-collapse (the low-bit-quant failure
    signature: 6-gram repetition >=4 or token-uniqueness <0.25). PASS/FAIL.

    gate="niah" (expensive, can take minutes at large `tokens`):
    `niah_certify()` — plants `needles` secret codes at spread depths in a
    `tokens`-long haystack, scores exact retrieval. All-or-nothing:
    `needles-1` of `needles` still reports status="error". `kv_type` is a
    caller-supplied label (not auto-detected) recorded alongside the score
    because it changes what the number means — an unlabelled score is a
    claim, not a certificate, per the function's own docstring.

    Both gates call the live gb-synapse proxy directly (whatever is
    currently served on GB_SYNAPSE_PORT) — serve the model to gate FIRST."""
    try:
        import gb_aviary
        if gate == "smoke":
            return gb_aviary.smoke_gate(model)
        if gate == "niah":
            return gb_aviary.niah_certify(model, tokens, needles=needles,
                                          kv_type=kv_type, seed=seed)
        return {"error": f"unknown gate '{gate}' — use 'smoke' or 'niah'"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(annotations=_READ_ONLY)
def synapse_resolve_preset() -> dict:
    """Which serving/presets/*.yaml preset is currently eligible and
    highest-priority, given live facts (NemoClaw audit, Phase 7) — kmod/
    shim/port observations (gb_readiness.py), host GPU compute capability
    (gb_topology.py), and online feeder count + per-feeder GPU cc
    (gb_cluster.py). Read-only: reports which preset WOULD apply, never
    changes what synapse_serve actually does — that live-dispatch wiring
    is deliberately deferred (see gb_synapse.resolve_serving_preset()'s
    docstring). On an ambiguous tie between two equal-priority eligible
    presets, returns `{"selected": null, "error": ..., "tied_preset_ids":
    [...]}` rather than silently picking one."""
    try:
        import gb_synapse
        return gb_synapse.resolve_serving_preset()
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
