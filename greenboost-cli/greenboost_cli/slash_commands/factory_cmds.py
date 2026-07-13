"""AI Factory management: /factory [start|stop|status|submit|agents|history]."""
from __future__ import annotations

import json

from greenboost_cli.terminal.commands import register_command


def _factory(args: str, session, settings: dict) -> None:
    parts = args.strip().split(None, 1)
    sub   = parts[0].lower() if parts else "status"
    rest  = parts[1] if len(parts) > 1 else ""

    dispatch = {
        "start":   _start,
        "stop":    _stop,
        "status":  _status,
        "submit":  _submit,
        "agents":  _agents,
        "pause":   _pause,
        "resume":  _resume,
        "history": _history,
        "help":    _help,
    }
    fn = dispatch.get(sub, _help)
    fn(rest, settings)


# ── Subcommands ───────────────────────────────────────────────────────────────

def _start(args: str, settings: dict) -> None:
    workers = 2
    try:
        if args.strip().isdigit():
            workers = max(1, min(int(args.strip()), 8))
    except ValueError:
        pass

    from greenboost_cli.workflow.factory import get_factory
    factory = get_factory()
    if factory._active:
        print("  Factory already running.")
        return

    # Add a default agent if none exist
    if not factory._agents:
        model = settings.get("model", "claude-sonnet-4-6")
        factory.add_agent("default", model=model)
        print(f"  Added default agent ({model})")

    factory.start(workers=workers)
    print(f"  AI Factory started — {workers} worker(s), {len(factory._agents)} agent(s)")
    print(f"  Dashboard: http://127.0.0.1:7821  (run: gb dashboard)")


def _stop(_args: str, _settings: dict) -> None:
    from greenboost_cli.workflow.factory import get_factory
    factory = get_factory()
    factory.stop()
    print("  AI Factory stopped.")


def _status(_args: str, _settings: dict) -> None:
    from greenboost_cli.workflow.factory import get_factory
    snap = get_factory().snapshot()
    state = "RUNNING" if snap["active"] else "stopped"
    print(f"  Factory  : {state}")
    print(f"  Queue    : {snap['queue_depth']} pending")
    print(f"  GPU      : {snap['gpu_ratio']:.1f}%")
    print()
    if snap["agents"]:
        print("  Agents:")
        for name, a in snap["agents"].items():
            st  = "PAUSED" if a["paused"] else a["current_task"]
            mem = f"{a['vram_used_mb']}+{a['vram_free_mb']}MB"
            ok  = a["total_tasks"]
            err = a["failed_tasks"]
            print(f"    {name:<16} {st:<35} vram={mem} ok={ok} fail={err}")
    else:
        print("  No agents registered. Use: /factory start")
    print()
    if snap["recent"]:
        print("  Recent completions:")
        for r in snap["recent"][:5]:
            mark = "✓" if r.get("success") else "✗"
            ag   = r.get("agent_name", "?")
            task = (r.get("prompt") or "")[:50]
            print(f"    {mark} {ag:<12} {task}")
    print()


def _submit(args: str, settings: dict) -> None:
    if not args.strip():
        print("  Usage: /factory submit <task description>")
        print("  Example: /factory submit write unit tests for the auth module")
        return

    from greenboost_cli.workflow.factory import get_factory
    factory = get_factory()
    if not factory._active:
        print("  Factory not running. Start it first: /factory start")
        return

    task_id = factory.submit(args.strip())
    print(f"  Task submitted — ID: {task_id}")
    print(f"  Queue depth: {factory._task_q.qsize()}")


def _agents(args: str, settings: dict) -> None:
    """Add, list, pause, resume agents."""
    parts = args.strip().split(None, 2)
    sub   = parts[0].lower() if parts else "list"

    from greenboost_cli.workflow.factory import get_factory
    factory = get_factory()

    if sub == "list" or sub == "":
        snap = get_factory().snapshot()
        if not snap["agents"]:
            print("  No agents. Add one: /factory agents add <name> [model]")
            return
        for name, a in snap["agents"].items():
            print(f"  {name:<16} model={a['model']:<24} task={a['current_task'][:30]}")

    elif sub == "add":
        name  = parts[1] if len(parts) > 1 else "agent1"
        model = parts[2] if len(parts) > 2 else settings.get("model", "claude-sonnet-4-6")
        factory.add_agent(name, model=model)
        print(f"  Agent '{name}' added (model: {model})")

    elif sub == "remove" and len(parts) > 1:
        factory.remove_agent(parts[1])
        print(f"  Agent '{parts[1]}' removed.")

    else:
        print("  /factory agents list")
        print("  /factory agents add <name> [model]")
        print("  /factory agents remove <name>")


def _pause(args: str, _settings: dict) -> None:
    name = args.strip()
    if not name:
        print("  Usage: /factory pause <agent_name>"); return
    from greenboost_cli.workflow.factory import get_factory
    get_factory().pause_agent(name)
    print(f"  Agent '{name}' paused.")


def _resume(args: str, _settings: dict) -> None:
    name = args.strip()
    if not name:
        print("  Usage: /factory resume <agent_name>"); return
    from greenboost_cli.workflow.factory import get_factory
    get_factory().resume_agent(name)
    print(f"  Agent '{name}' resumed.")


def _history(_args: str, _settings: dict) -> None:
    from greenboost_cli.workflow.factory import get_factory
    rows = get_factory().db.recent_completions(20)
    if not rows:
        print("  No completed tasks yet."); return
    print(f"  Recent completions ({len(rows)}):")
    for r in rows:
        mark = "✓" if r.get("success") else "✗"
        dur  = ""
        if r.get("finished_at") and r.get("started_at"):
            dur = f" {r['finished_at']-r['started_at']:.1f}s"
        print(f"  {mark} {r.get('agent_name','?'):<12}{dur:>7}  {(r.get('prompt') or '')[:60]}")
    print()


def _help(_args: str, _settings: dict) -> None:
    print("  /factory start [workers]    Start factory (default 2 workers)")
    print("  /factory stop               Stop factory")
    print("  /factory status             Current state (agents, queue, GPU)")
    print("  /factory submit <task>      Submit a task")
    print("  /factory agents [add|remove|list]")
    print("  /factory pause <agent>      Pause an agent")
    print("  /factory resume <agent>     Resume a paused agent")
    print("  /factory history            Recent completed tasks")


# ── Registration ──────────────────────────────────────────────────────────────

def register(settings: dict) -> None:
    register_command(
        "factory",
        lambda args, session: _factory(args, session, settings),
        description="AI factory multi-agent orchestration",
    )
