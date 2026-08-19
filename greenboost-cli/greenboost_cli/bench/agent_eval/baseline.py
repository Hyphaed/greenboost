"""The baseline that already exists: read it out of dataflux.

AE-1 originally proposed standing up a synthetic benchmark before touching
anything. That was half wrong. GreenBoost has been recording its own agent
sessions all along , that is what the flight recorder is FOR , so the
retrospective baseline is free, real, and not subject to "the benchmark isn't
representative".

What the 14-day window said when this module was written (n=32 turns):

    prompt-cache hit  >=99%   median TTFT     5.5s
                      90-99%  median TTFT    33.4s
                      50-90%  median TTFT   140.3s
                      <50%    median TTFT   165.9s   (max 553.8s)

    44.5 minutes of time-to-first-token across 32 turns; median 35.5s per
    turn spent producing nothing. 4 of 32 turns reused no prefix at all.

That is the AE-2 argument in one table: TTFT is ~30x worse at a cold prefix
than a warm one, and every history rewrite makes the prefix cold. No paper
required.

Tool mix over the same window (n=124): Bash 102, Write 16, Read 4,
TodoWrite 1, Skill 1, MCP 0. The MCP surface costs context on every request
and was called zero times.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict, field

#: Cache-hit buckets, in the order they are reported.
_BUCKETS = (
    (">=99%", 99.0),
    ("90-99%", 90.0),
    ("50-90%", 50.0),
    ("<50%", 0.0),
)


@dataclass
class CacheBaseline:
    n_turns: int = 0
    median_hit_pct: float = 0.0
    zero_hit_turns: int = 0
    median_ttft_s: float = 0.0
    total_ttft_min: float = 0.0
    by_bucket: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolBaseline:
    n_calls: int = 0
    by_name: dict = field(default_factory=dict)
    by_decision: dict = field(default_factory=dict)
    mcp_calls: int = 0
    mcp_share: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _bucket_for(hit_pct: float) -> str:
    for name, floor in _BUCKETS:
        if hit_pct >= floor:
            return name
    return _BUCKETS[-1][0]


def cache_baseline(events) -> CacheBaseline:
    """Prompt-cache reuse vs time-to-first-token, from `prompt_cache` events."""
    pc = [e for e in events if e.get("kind") == "prompt_cache"]
    out = CacheBaseline(n_turns=len(pc))
    if not pc:
        return out
    hits = [float(e.get("hit_pct") or 0.0) for e in pc]
    ttfts = [float(e.get("ttft_ms") or 0.0) / 1000.0 for e in pc]
    out.median_hit_pct = round(statistics.median(hits), 1)
    out.zero_hit_turns = sum(1 for h in hits if h <= 0.0)
    out.median_ttft_s = round(statistics.median(ttfts), 1)
    out.total_ttft_min = round(sum(ttfts) / 60.0, 1)
    grouped: dict = {}
    for h, t in zip(hits, ttfts):
        grouped.setdefault(_bucket_for(h), []).append(t)
    out.by_bucket = {
        name: {
            "n": len(grouped[name]),
            "median_ttft_s": round(statistics.median(grouped[name]), 1),
            "max_ttft_s": round(max(grouped[name]), 1),
        }
        for name, _ in _BUCKETS if name in grouped
    }
    return out


def tool_baseline(events) -> ToolBaseline:
    """What the agent actually reached for, from `cli_tool_call` events."""
    tc = [e for e in events if e.get("kind") == "cli_tool_call"]
    out = ToolBaseline(n_calls=len(tc))
    if not tc:
        return out
    for e in tc:
        name = e.get("name") or "(unnamed)"
        out.by_name[name] = out.by_name.get(name, 0) + 1
        dec = e.get("decision") or e.get("status") or "(none)"
        out.by_decision[dec] = out.by_decision.get(dec, 0) + 1
    out.mcp_calls = sum(c for n, c in out.by_name.items() if n.startswith("mcp__"))
    out.mcp_share = round(out.mcp_calls / len(tc), 3)
    return out


def read_baseline(days: float = 14.0) -> dict:
    """Both baselines over the last `days`, straight from the flight recorder."""
    try:
        import gb_dataflux
    except ImportError:
        return {"error": "gb_dataflux is not importable , is GreenBoost installed?"}
    events = gb_dataflux.read_events(since_hours=days * 24.0)
    return {
        "days": days,
        "events_scanned": len(events),
        "cache": cache_baseline(events).to_dict(),
        "tools": tool_baseline(events).to_dict(),
    }
