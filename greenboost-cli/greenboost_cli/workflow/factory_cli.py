"""Headless `gb factory-*` subcommands.

These handlers expose the AIFactory to scripted callers (e.g. optimal-claude's
gb_bridge). Each handler is purely process-local — there is no new network
listener — and all user input is normalised through greenboost_cli.security
before reaching the factory.

Subcommands:
  gb factory-submit "<prompt>" [--agent NAME] [--priority N]
                               [--metadata-json JSON] [--json]
  gb factory-status            [--json]
  gb factory-list              [--state pending|running|completed|all]
                               [--agent NAME] [--limit N] [--json]
  gb factory-pause   <agent>   [--json]
  gb factory-resume  <agent>   [--json]
  gb factory-agents            [--json]
  gb factory-hot-swap <agent> <old_skill> <new_skill> [--json]
  gb factory-sleep   on|off    [--agent NAME] [--json]

Every handler returns an int exit code (0 = success).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from greenboost_cli.security import (
    _cap, _safe_project,
    _MAX_QUERY_LEN, _MAX_TEXT_LEN, _MAX_PROJECT_LEN,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _emit_json(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


def _emit_err(msg: str) -> None:
    sys.stderr.write(f"gb: {msg}\n")


def _safe_agent_name(name: str) -> str:
    """Reuse the project-name validator: same rules (no path separators / control chars)."""
    return _safe_project(name)


def _safe_skill_name(name: str) -> str:
    """Skill names are also stored as plain identifiers."""
    return _safe_project(name)


def _get_factory():
    from greenboost_cli.workflow.factory import get_factory
    return get_factory()


# ── factory-submit ────────────────────────────────────────────────────────────

def cmd_factory_submit(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb factory-submit", add_help=True)
    p.add_argument("prompt")
    p.add_argument("--agent", default="")
    p.add_argument("--priority", type=int, default=10)
    p.add_argument("--autonomous", action="store_true",
                   help="Mark as autonomous (no permission prompts).")
    p.add_argument("--metadata-json", default="",
                   help="JSON object merged into the task metadata.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    prompt = _cap(args.prompt, _MAX_QUERY_LEN)
    agent = _safe_agent_name(args.agent) if args.agent else ""
    priority = max(1, min(int(args.priority), 100))

    metadata: dict = {}
    if args.metadata_json:
        raw = _cap(args.metadata_json, _MAX_TEXT_LEN)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                metadata = parsed
            else:
                _emit_err("--metadata-json must be a JSON object")
                return 2
        except json.JSONDecodeError as e:
            _emit_err(f"invalid --metadata-json: {e}")
            return 2

    factory = _get_factory()
    task_id = factory.submit(
        prompt=prompt,
        agent_name=agent,
        priority=priority,
        autonomous=bool(args.autonomous),
        metadata=metadata,
    )
    if args.json:
        _emit_json({
            "task_id":    task_id,
            "agent":      agent,
            "priority":   priority,
            "queue_depth": factory._task_q.qsize(),
        })
    else:
        if task_id:
            print(f"  task_id   : {task_id}")
            print(f"  agent     : {agent or '(auto)'}")
            print(f"  priority  : {priority}")
            print(f"  queued    : {factory._task_q.qsize()} total")
        else:
            print("  rejected (delegation depth exceeded or invalid prompt)")
    return 0 if task_id else 1


# ── factory-status ────────────────────────────────────────────────────────────

def cmd_factory_status(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb factory-status", add_help=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    factory = _get_factory()
    snap = factory.snapshot()

    if args.json:
        _emit_json(snap)
        return 0

    print(f"  factory   : {'RUNNING' if snap['active'] else 'stopped'}")
    print(f"  queue     : {snap['queue_depth']} pending")
    print(f"  gpu       : {snap['gpu_ratio']:.1f}%")
    print(f"  sleep     : default={snap.get('sleep_default', False)}")
    print()
    agents = snap.get("agents", {})
    if not agents:
        print("  (no agents registered)")
    else:
        print("  agents:")
        for name, a in agents.items():
            tag = "PAUSED" if a["paused"] else a["current_task"]
            sl  = "z" if a.get("sleep_enabled") else " "
            sk  = ",".join(a.get("skills", [])) or "-"
            print(f"    [{sl}] {name:<14} {tag[:30]:<30} skills={sk}")
    print()
    if snap.get("recent"):
        print("  recent:")
        for r in snap["recent"][:5]:
            mark = "ok" if r.get("success") else "FAIL"
            print(f"    {mark:<4} {r.get('agent_name','?'):<12} "
                  f"{(r.get('prompt') or '')[:50]}")
    return 0


# ── factory-list ──────────────────────────────────────────────────────────────

def cmd_factory_list(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb factory-list", add_help=True)
    p.add_argument("--state", default="all",
                   choices=["pending", "running", "completed", "all"])
    p.add_argument("--agent", default=None)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    limit = max(1, min(int(args.limit), 500))
    agent = _safe_agent_name(args.agent) if args.agent else None

    factory = _get_factory()
    data = factory.list_tasks(state=args.state, limit=limit, agent=agent)

    if args.json:
        _emit_json(data)
        return 0

    if data["state"] in ("pending", "all"):
        print(f"  pending ({len(data['pending'])}):")
        for r in data["pending"]:
            print(f"    p={r['priority']:<3} {r['task_id']:<14} "
                  f"agent={r['agent_name'] or '(auto)':<10} "
                  f"{r['prompt'][:60]}")
    if data["state"] in ("running", "all"):
        print(f"  running ({len(data['running'])}):")
        for r in data["running"]:
            print(f"    {r['task_id']:<14} agent={r['agent']:<10} "
                  f"{r['prompt'][:60]}")
    if data["state"] in ("completed", "all"):
        print(f"  completed ({len(data['completed'])}):")
        for r in data["completed"]:
            mark = "ok" if r.get("success") else "FAIL"
            print(f"    {mark:<4} {r.get('task_id','?'):<14} "
                  f"agent={r.get('agent_name','?'):<10} "
                  f"{(r.get('prompt') or '')[:50]}")
    return 0


# ── factory-pause / factory-resume ────────────────────────────────────────────

def _toggle_agent(argv: list[str], action: str) -> int:
    p = argparse.ArgumentParser(
        prog=f"gb factory-{action}", add_help=True,
    )
    p.add_argument("agent")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    agent = _safe_agent_name(args.agent)
    if not agent:
        _emit_err("agent name required")
        return 2

    factory = _get_factory()
    if agent not in factory._agents:
        _emit_err(f"agent '{agent}' not found")
        if args.json:
            _emit_json({"status": "error",
                        "error": f"agent '{agent}' not found"})
        return 3

    if action == "pause":
        factory.pause_agent(agent)
    else:
        factory.resume_agent(agent)

    if args.json:
        _emit_json({"status": "ok", "agent": agent, "action": action})
    else:
        print(f"  agent '{agent}' {action}d")
    return 0


def cmd_factory_pause(argv: list[str]) -> int:
    return _toggle_agent(argv, "pause")


def cmd_factory_resume(argv: list[str]) -> int:
    return _toggle_agent(argv, "resume")


# ── factory-agents ────────────────────────────────────────────────────────────

def cmd_factory_agents(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb factory-agents", add_help=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    factory = _get_factory()
    agents = factory.list_agents()

    if args.json:
        _emit_json({"agents": agents, "count": len(agents)})
        return 0

    if not agents:
        print("  (no agents)")
        return 0
    for a in agents:
        flags = []
        if a["paused"]:        flags.append("paused")
        if a["sleep_enabled"]: flags.append("sleep")
        ftxt = ",".join(flags) if flags else "-"
        cur  = a["current_task"]
        if a["current_task_id"]:
            cur = f"{cur} ({a['current_task_id']})"
        sk = ",".join(a["skills"]) if a["skills"] else "-"
        print(f"  {a['name']:<14} model={a['model']:<22} flags={ftxt:<14} "
              f"idle={a['idle_seconds']:>6}s  vram={a['vram_used_mb']}MB")
        print(f"      task   : {cur[:70]}")
        print(f"      skills : {sk}")
        if a.get("pending_swap"):
            print(f"      pending: swap {a['pending_swap'][0]} -> {a['pending_swap'][1]}")
    return 0


# ── factory-hot-swap ──────────────────────────────────────────────────────────

def cmd_factory_hot_swap(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb factory-hot-swap", add_help=True)
    p.add_argument("agent")
    p.add_argument("old_skill")
    p.add_argument("new_skill")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    agent = _safe_agent_name(args.agent)
    old_skill = _safe_skill_name(args.old_skill)
    new_skill = _safe_skill_name(args.new_skill)

    if not (agent and old_skill and new_skill):
        _emit_err("agent, old_skill, new_skill all required")
        return 2

    factory = _get_factory()
    result = factory.hot_swap_skill(agent, old_skill, new_skill)

    if args.json:
        _emit_json(result)
    else:
        st = result.get("status", "?")
        if st == "applied":
            print(f"  swap applied : {old_skill} -> {new_skill} on {agent}")
            print(f"  skills now   : {','.join(result.get('skills', []))}")
        elif st == "queued":
            print(f"  swap queued  : {old_skill} -> {new_skill} on {agent}")
            print( "  (will apply when current task finishes)")
        else:
            print(f"  ERROR        : {result.get('error', 'unknown')}")
    return 0 if result.get("status") in ("applied", "queued") else 1


# ── factory-sleep ─────────────────────────────────────────────────────────────

def cmd_factory_sleep(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb factory-sleep", add_help=True)
    p.add_argument("mode", choices=["on", "off"])
    p.add_argument("--agent", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    enabled = (args.mode == "on")
    agent   = _safe_agent_name(args.agent) if args.agent else None

    factory = _get_factory()
    result = factory.set_sleep(enabled=enabled, agent_name=agent)

    if args.json:
        _emit_json(result)
    else:
        if result.get("status") == "ok":
            scope = f"agent={agent}" if agent else "all agents"
            print(f"  sleep mode {'ON' if enabled else 'OFF'} for {scope}")
            if result.get("changed"):
                print(f"  changed: {', '.join(result['changed'])}")
        else:
            print(f"  ERROR: {result.get('error', 'unknown')}")
    return 0 if result.get("status") == "ok" else 1


# ── Dispatch registration helpers ─────────────────────────────────────────────

FACTORY_SUBCOMMANDS = {
    "factory-submit":   cmd_factory_submit,
    "factory-status":   cmd_factory_status,
    "factory-list":     cmd_factory_list,
    "factory-pause":    cmd_factory_pause,
    "factory-resume":   cmd_factory_resume,
    "factory-agents":   cmd_factory_agents,
    "factory-hot-swap": cmd_factory_hot_swap,
    "factory-sleep":    cmd_factory_sleep,
}


__all__ = [
    "FACTORY_SUBCOMMANDS",
    "cmd_factory_submit",
    "cmd_factory_status",
    "cmd_factory_list",
    "cmd_factory_pause",
    "cmd_factory_resume",
    "cmd_factory_agents",
    "cmd_factory_hot_swap",
    "cmd_factory_sleep",
]
