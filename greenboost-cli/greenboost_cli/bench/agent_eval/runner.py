"""Run the task set against the real served model and score it.

The retrospective half of AE-1 lives in baseline.py and needs no model. This
half is the controlled A/B: same tasks, same model, before and after a change.

Deliberately drives `run_subagent`, not a private code path , the benchmark
must exercise the turn loop the user actually gets, including dispatch,
approval and the concurrency partitioner. A benchmark with its own execution
path measures its own execution path.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from greenboost_cli.bench.agent_eval.scoring import score_task, aggregate, TaskScore
from greenboost_cli.bench.agent_eval.tasks import DEFAULT_TASKS


def _requirements_met(task, settings) -> str:
    """Empty string when the task can run; otherwise the skip reason."""
    if "mcp" in task.requires:
        try:
            from greenboost_cli.mcp.client import discover_mcp_json
            if discover_mcp_json() is None:
                return "no .mcp.json discovered"
        except Exception as e:
            return f"MCP unavailable: {type(e).__name__}"
    return ""


def run_task(task, settings=None, timeout_s: float = 900.0) -> TaskScore:
    skip = _requirements_met(task, settings)
    if skip:
        s = TaskScore(task_id=task.id)
        s.skipped_reason = skip
        return s
    from greenboost_cli.agents.subagent import run_subagent
    r = run_subagent(task.prompt, label=f"eval:{task.id}",
                     timeout_s=timeout_s, settings=settings)
    trace = {
        "summary": r.summary,
        "tool_calls": [{"name": c.name, "result": c.result} for c in r.tool_calls],
        "tokens": r.tokens_used,
        "duration_s": r.duration_s,
        "error": r.error or ("timed out" if r.timed_out else ""),
    }
    return score_task(task, trace)


def run_eval(tasks=None, settings=None, out_path=None, timeout_s: float = 900.0) -> dict:
    """Run every task, aggregate, emit `agent_eval_run` to dataflux, return the report."""
    tasks = tasks or DEFAULT_TASKS
    started = time.time()
    scores = [run_task(t, settings=settings, timeout_s=timeout_s) for t in tasks]
    run = aggregate(scores)
    report = {
        "started_ts": started,
        "wall_s": round(time.time() - started, 1),
        "model": (settings or {}).get("model", ""),
        "run": run.to_dict(),
    }
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "agent_eval_run",
            "status": "ok",
            "n_items": run.tasks_run,
            "overall": round(run.overall, 3),
            "completion": round(run.completion, 3),
            "tool_selection": round(run.tool_selection, 3),
            "efficiency": round(run.efficiency, 3),
            "grounding": round(run.grounding, 3),
            "total_tokens": run.total_tokens,
            "duration_s": report["wall_s"],
            "model": report["model"],
        })
    except Exception:
        pass                      # telemetry must never fail a benchmark run
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="agent_eval")
    ap.add_argument("--baseline", action="store_true",
                    help="print the retrospective dataflux baseline and exit "
                         "(no model needed)")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--out", default="", help="write the JSON report here")
    a = ap.parse_args(argv)
    if a.baseline:
        from greenboost_cli.bench.agent_eval.baseline import read_baseline
        print(json.dumps(read_baseline(a.days), indent=2))
        return 0
    from greenboost_cli.environment.settings import load_settings
    print(json.dumps(run_eval(settings=load_settings(), out_path=a.out or None), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
