"""gb_actuation , shared gated-actuation verbs for the agent surface.

ONE implementation of every write action an LLM/agent can drive, called by
BOTH the MCP tools (gb_cluster_mcp / gb_mcp / gb_synapse_mcp) and the A2A
gateway (gb_a2a). No duplicated actuation logic between control planes.

Safety model (identical to gb_mcp.optimize_inference): every verb is DRY-RUN
(returns the plan it *would* apply) unless BOTH:
  - the caller passes confirm=True, AND
  - the server env has GB_ORCH_ACTUATE=1
When applied, the verb emits a dataflux `actuation` event so it is auditable in
the flight recorder and correlatable with the snapshot that triggered it.

Only remotely-meaningful actuations live here. Actions that need an in-process
object (quantize_to_fit(model), ModelTierManager.promote(name)) are NOT exposed
, an agent can't hand a live torch model over MCP. The actuatable equivalents
are policy/lever writes (quant budget+quality env, GbControl tier levers) that
DO steer the next pipeline run.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

# Shared env file ai-forge / pipelines read at process start (FORGE_OLLAMA_URL,
# GB_QUANT_BUDGET_GB, GB_QUALITY, ...). Env-overridable; never a frozen literal.
INFERENCE_ENV = Path(os.environ.get("GB_INFERENCE_ENV", "/etc/greenboost/inference.env"))


# ── gate + audit ────────────────────────────────────────────────────────────

def actuation_gate(confirm: bool) -> dict:
    """Return {"allowed": bool, "reason": str}. Allowed only when confirm=True
    AND GB_ORCH_ACTUATE=1 (the double gate). A2A cannot bypass either half."""
    if not confirm:
        return {"allowed": False, "reason": "dry-run (confirm=False)"}
    if os.environ.get("GB_ORCH_ACTUATE") != "1":
        return {"allowed": False, "reason": "GB_ORCH_ACTUATE!=1 (double gate)"}
    return {"allowed": True, "reason": "confirm=True and GB_ORCH_ACTUATE=1"}


def _emit(verb: str, gated: bool, **fields) -> None:
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "actuation", "kind": "actuation",
            "stage": verb, "lever": verb, "gated": gated, "status": "ok",
            "n_items": 0, "items": [], "duration_s": 0.0,
            **{k: v for k, v in fields.items()
               if isinstance(v, (int, float, str, bool))},
        })
    except Exception:
        pass


def _read_env_file() -> dict:
    out: dict[str, str] = {}
    try:
        for line in INFERENCE_ENV.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def _write_env_file(updates: dict) -> bool:
    """Merge `updates` into INFERENCE_ENV (create dir if needed). Best-effort;
    returns True on success."""
    try:
        env = _read_env_file()
        env.update({k: str(v) for k, v in updates.items()})
        INFERENCE_ENV.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(f"{k}={v}\n" for k, v in sorted(env.items()))
        tmp = INFERENCE_ENV.with_suffix(".tmp")
        tmp.write_text("# Written by gb_actuation , GreenBoost inference policy\n" + body)
        tmp.rename(INFERENCE_ENV)
        return True
    except OSError:
        return False


# ── verbs ───────────────────────────────────────────────────────────────────

def cluster_ensure_feeder_ready(feeder_ip: str, confirm: bool = False) -> dict:
    """Provision + verify a feeder (rsync code, ensure model, import-check deps)
    so a visibly-idle feeder can be put to work. Wraps
    gb_cluster.ensure_feeder_ready , adds no dispatch logic."""
    gate = actuation_gate(confirm)
    plan = {"verb": "cluster_ensure_feeder_ready", "feeder_ip": feeder_ip,
            "gate": gate}
    if not gate["allowed"]:
        plan["dry_run"] = f"would provision+verify feeder {feeder_ip}"
        return plan
    try:
        import gb_cluster
        target = next((f for f in gb_cluster.feeders(probe=True)
                       if f.ip == feeder_ip), None)
        if target is None:
            plan["error"] = f"no configured feeder with ip {feeder_ip}"
            return plan
        ok = gb_cluster.ensure_feeder_ready(target)
        plan["applied"] = bool(ok)
        _emit("cluster_ensure_feeder_ready", True, feeder_ip=feeder_ip, ok=bool(ok))
    except Exception as e:
        plan["error"] = str(e)
    return plan


def cluster_dispatch_plan(confirm: bool = False) -> dict:
    """Report which online feeders are dispatch-eligible right now (not busy,
    link fast enough) and , when gated , provision them so the next pipeline
    dispatch fans out. Actual item dispatch needs the caller's own
    run_local/run_remote callables (cluster_map), so this verb prepares the
    cluster; it does not fabricate work."""
    gate = actuation_gate(confirm)
    plan = {"verb": "cluster_dispatch_plan", "gate": gate}
    try:
        import gb_cluster
        snap = gb_cluster.cluster_snapshot()
        eligible = [f for f in snap.get("feeders", [])
                    if f.get("online") and f.get("gpu_util_pct", 0) < 85]
        plan["eligible_feeders"] = [f.get("hostname") or f.get("ip") for f in eligible]
        if not gate["allowed"]:
            plan["dry_run"] = f"would provision {len(eligible)} feeder(s) for dispatch"
            return plan
        provisioned = []
        for f in eligible:
            r = cluster_ensure_feeder_ready(f.get("ip", ""), confirm=True)
            if r.get("applied"):
                provisioned.append(f.get("hostname") or f.get("ip"))
        plan["provisioned"] = provisioned
        _emit("cluster_dispatch_plan", True, provisioned=len(provisioned))
    except Exception as e:
        plan["error"] = str(e)
    return plan


def set_quant_policy(budget_gb: float | None = None, quality: str | None = None,
                     confirm: bool = False) -> dict:
    """Set the quant policy the NEXT pipeline run reads: GB_QUANT_BUDGET_GB and
    GB_QUALITY in the shared inference.env. Enforces the fp8 quality floor ,
    quality below fp8 (nvfp4/int8/int4/tq*) requires an explicit non-default
    value AND is surfaced as a tradeoff, never silently applied."""
    gate = actuation_gate(confirm)
    updates: dict[str, str] = {}
    if budget_gb is not None:
        updates["GB_QUANT_BUDGET_GB"] = f"{float(budget_gb):.1f}"
    tradeoff = None
    if quality is not None:
        below_fp8 = {"nvfp4", "8", "int8", "4", "int4", "tq3", "tq2"}
        if str(quality).lower() in below_fp8:
            tradeoff = (f"quality={quality} is below the fp8 floor , accept only "
                        f"with a measured quality gate (niah_certify/smoke_gate)")
        updates["GB_QUALITY"] = str(quality)
    plan = {"verb": "set_quant_policy", "gate": gate, "updates": updates,
            "env_file": str(INFERENCE_ENV)}
    if tradeoff:
        plan["fp8_floor_tradeoff"] = tradeoff
    if not gate["allowed"]:
        plan["dry_run"] = f"would write {updates} to {INFERENCE_ENV}"
        return plan
    plan["applied"] = _write_env_file(updates)
    _emit("set_quant_policy", True, **updates)
    return plan


# GbControl lever name → (setter method, value coercion, unit label)
_TIER_LEVERS = {
    "kv_reserve_mb":         ("set_kv_reserve_mb", int, "MB"),
    "safety_reserve_gb":     ("set_safety_reserve_gb", int, "GB"),
    "workstation_reserve_mb":("set_workstation_reserve_mb", int, "MB"),
    "virtual_vram_gb":       ("set_virtual_vram_gb", int, "GB"),
    "pool_cap_mb":           ("set_pool_cap_mb", int, "MB"),
}


def tier_actuate(lever: str, value: int, confirm: bool = False) -> dict:
    """Move a GB-Tiering lever via GbControl: one of kv_reserve_mb,
    safety_reserve_gb, workstation_reserve_mb, virtual_vram_gb, pool_cap_mb.
    (Per-buffer promote/demote/evict is in-process only , not remotable.)"""
    gate = actuation_gate(confirm)
    plan = {"verb": "tier_actuate", "lever": lever, "value": value, "gate": gate,
            "valid_levers": sorted(_TIER_LEVERS)}
    if lever not in _TIER_LEVERS:
        plan["error"] = f"unknown lever {lever!r}"
        return plan
    if not gate["allowed"]:
        plan["dry_run"] = f"would set {lever}={value} via GbControl"
        return plan
    try:
        from gb_control import GbControl
        method, coerce, _unit = _TIER_LEVERS[lever]
        res = getattr(GbControl(), method)(coerce(value), reason="mcp/a2a tier_actuate")
        plan["applied"] = getattr(res, "ok", None)
        plan["result"] = getattr(res, "reason", str(res))
        _emit("tier_actuate", True, lever=lever, value=value)
    except Exception as e:
        plan["error"] = str(e)
    return plan


def serve_and_repoint(model: str, port: int = 11434,
                      forge_url_target: str | None = None,
                      confirm: bool = False) -> dict:
    """One step for the "prefer gb-synapse" rule: serve `model` via gb-synapse
    AND repoint FORGE_OLLAMA_URL (in inference.env) so ai-forge pipelines use
    the gb-synapse proxy instead of raw ollama. Closes the two-control-plane
    gap (serve on one server + out-of-band env edit) into a single actuation."""
    gate = actuation_gate(confirm)
    target = forge_url_target or f"127.0.0.1:{port}"
    plan = {"verb": "serve_and_repoint", "model": model, "port": port,
            "forge_url_target": target, "gate": gate}
    if not gate["allowed"]:
        plan["dry_run"] = (f"would serve {model!r} via gb-synapse:{port} and set "
                           f"FORGE_OLLAMA_URL={target}")
        return plan
    try:
        import gb_synapse
        state = gb_synapse.serve(model, port=port)
        plan["served"] = getattr(state, "as_dict", lambda: str(state))()
        # Never point FORGE_OLLAMA_URL at a proxy that isn't actually
        # answering — every ai-forge pipeline reading that URL would start
        # hitting a dead endpoint. proxy_error (added 2026-07-16) is the
        # real signal; a stale ServerState from before that field existed
        # has no attribute at all, hence the getattr default of None (never
        # block on an old field that can't exist yet).
        proxy_error = getattr(state, "proxy_error", None)
        if proxy_error:
            plan["repointed"] = False
            plan["error"] = f"engine served but proxy did not come up — NOT repointing: {proxy_error}"
        else:
            plan["repointed"] = _write_env_file({"FORGE_OLLAMA_URL": target})
            _emit("serve_and_repoint", True, model=model, forge_url=target)
    except Exception as e:
        plan["error"] = str(e)
    return plan


def shim_env(workload: str = "diffusion", enabled: bool = True) -> dict:
    """QUERY-only (no gate): the LD_PRELOAD env overlay that turns GreenBoost on
    for a subprocess of `workload` type. Lets an agent obtain the exact env to
    launch a GreenBoost-accelerated run without a Python import."""
    try:
        import gb_cluster
        env = gb_cluster.shim_env(workload=workload, enabled=enabled, base_env={})
        return {"verb": "shim_env", "workload": workload, "enabled": enabled,
                "env": env}
    except Exception as e:
        return {"verb": "shim_env", "error": str(e)}


def run_under_greenboost(command: "list[str] | str", workload: str = "llm",
                         cwd: str = "", timeout_s: int = 900,
                         confirm: bool = False) -> dict:
    """A8: closes "discover -> configure -> execute -> observe" entirely
    through MCP/A2A, without a separate shell step. Runs `command` as a
    subprocess with the `shim_env(workload)` overlay applied.

    SECURITY NOTE — the one verb here that executes a command, unlike every
    other verb (which only writes a policy/env value): `command` is ALWAYS
    executed as an argv list (`shell=False`; a string is tokenized with
    shlex, never handed to a shell), so pipes/redirects/globs/`&&` have no
    special meaning — this preserves gb_a2a's "no shell passthrough" rule
    even though this verb runs a real process. Still double-gated exactly
    like every other verb; dry-run returns the resolved argv + env overlay
    without executing anything.

    Emits kind="agent_run" — one event at start (status=started) and one at
    completion (status=ok/error, exit_code, duration_s) so every run is
    auditable in the flight recorder like any other actuation.
    """
    gate = actuation_gate(confirm)
    argv = command if isinstance(command, list) else shlex.split(command)
    run_id = f"run_{int(time.time() * 1000)}"
    plan = {"verb": "run_under_greenboost", "run_id": run_id, "command": argv,
            "workload": workload, "cwd": cwd or None, "timeout_s": timeout_s,
            "gate": gate}
    if not gate["allowed"]:
        plan["dry_run"] = f"would run {argv!r} under shim_env(workload={workload!r})"
        return plan
    try:
        import gb_cluster
        env_overlay = gb_cluster.shim_env(workload=workload, enabled=True, base_env={})
    except Exception as e:
        plan["error"] = f"shim_env failed: {e}"
        return plan

    run_env = {**os.environ, **env_overlay}
    _emit("run_under_greenboost", True, run_id=run_id, status="started",
          command=" ".join(argv), workload=workload)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=cwd or None, env=run_env, shell=False,
                              capture_output=True, text=True, timeout=timeout_s)
        duration = time.monotonic() - t0
        plan["exit_code"]   = proc.returncode
        plan["duration_s"]  = round(duration, 2)
        plan["stdout_tail"] = proc.stdout[-4000:]
        plan["stderr_tail"] = proc.stderr[-4000:]
        _emit("run_under_greenboost", True, run_id=run_id,
              status="ok" if proc.returncode == 0 else "error",
              exit_code=proc.returncode, duration_s=round(duration, 2))
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - t0
        plan["error"] = f"timed out after {timeout_s}s"
        plan["duration_s"] = round(duration, 2)
        _emit("run_under_greenboost", True, run_id=run_id, status="error",
              error="timeout", duration_s=round(duration, 2))
    except Exception as e:
        plan["error"] = str(e)
        _emit("run_under_greenboost", True, run_id=run_id, status="error", error=str(e))
    return plan


# ── AgentCard (for A2A) ───────────────────────────────────────────────────────

def agent_card(bind: str = "127.0.0.1:8790") -> dict:
    """A2A AgentCard: advertises this GreenBoost cluster's per-node hardware and
    the capability verbs an agent may invoke. Served at /.well-known/agent.json.
    """
    version = "unknown"
    nodes: list = []
    try:
        import gb_monitor
        version = gb_monitor.snapshot().as_dict().get("version", "unknown")
    except Exception:
        pass
    try:
        import gb_cluster
        topo = gb_cluster.cluster_topology()
        host = topo.get("host", {})
        nodes.append({"role": "host", **{k: host.get(k) for k in
                      ("gpu_name", "vram_gb", "ram_total_gb", "pcie_gen") if k in host}})
        for key, ft in (topo.get("feeders") or {}).items():
            nodes.append({"role": "feeder", "id": key,
                          "hostname": ft.get("hostname"), "online": ft.get("online"),
                          **{k: ft.get(k) for k in
                             ("gpu_name", "vram_gb", "ram_total_gb") if k in ft}})
    except Exception:
        pass
    return {
        "name": "greenboost-cluster",
        "description": ("GreenBoost unified-GPU cluster , observe and actuate "
                        "tiering, quant policy, cluster provisioning, and "
                        "gb-synapse serving. All actuation is double-gated."),
        "version": version,
        "url": f"http://{bind}/",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "skills": [
            {"id": v, "name": v, "description": (fn.__doc__ or "").strip().split("\n")[0]}
            for v, fn in VERBS.items()
        ],
        "nodes": nodes,
    }


def agent_card_v03(bind: str = "127.0.0.1:8790") -> dict:
    """A2A protocol v0.3-shaped AgentCard (camelCase field names per the A2A
    spec: https://google.github.io/A2A/), served at
    /.well-known/agent-card.json — interop with ai-forge's
    studio/server/a2a_gateway.py (a2a-sdk v0.3), which serves the same path.
    See docs/a2a-interop.md for the full two-gateway split rationale: this
    (gb_a2a.py) is the NODE-level actuation gateway (gated GbControl/cluster
    levers), ai-forge's is the STUDIO task-delegation gateway (generation
    jobs) — different concerns, same discovery convention. Same underlying
    data as agent_card(); a different JSON shape, not a different verb set —
    `run_under_greenboost` (JSON-RPC method) still dispatches through the
    SAME gb_actuation.VERBS table via the legacy /  POST endpoint."""
    legacy = agent_card(bind=bind)
    return {
        "protocolVersion": "0.3",
        "name": "greenboost-node",
        "description": legacy["description"],
        "url": legacy["url"],
        "version": legacy["version"],
        "preferredTransport": "JSONRPC",
        "capabilities": {"streaming": False,
                         "extensions": [{"uri": "urn:greenboost:node",
                                        "description": "GreenBoost node hardware + tier telemetry",
                                        "params": {"nodes": legacy["nodes"]}}]},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [{"id": s["id"], "name": s["name"], "description": s["description"],
                   "tags": ["greenboost", "gpu", "cluster"],
                   "inputModes": ["application/json"], "outputModes": ["application/json"]}
                  for s in legacy["skills"]],
        "provider": {"organization": "greenboost", "url": legacy["url"]},
    }


# Verb registry , the A2A JSON-RPC method table AND the MCP tool bodies both
# resolve through this, so the two control planes can never drift.
VERBS = {
    "cluster_ensure_feeder_ready": cluster_ensure_feeder_ready,
    "cluster_dispatch_plan": cluster_dispatch_plan,
    "set_quant_policy": set_quant_policy,
    "tier_actuate": tier_actuate,
    "serve_and_repoint": serve_and_repoint,
    "shim_env": shim_env,
    "run_under_greenboost": run_under_greenboost,
}
