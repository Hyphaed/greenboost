"""Per-project token usage tracking."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def _token_file(project_dir: Path) -> Path:
    return project_dir / "token_usage.json"


def _load(project_dir: Path) -> dict:
    f = _token_file(project_dir)
    if not f.exists():
        return {"sessions": [], "totals": {"api": 0, "local": 0}}
    with open(f) as fh:
        return json.load(fh)


def _save(project_dir: Path, data: dict) -> None:
    with open(_token_file(project_dir), "w") as fh:
        json.dump(data, fh, indent=2)


def record(project_dir: Path, api_tokens: int, local_tokens: int, session_id: str = "",
           tok_s: float = 0.0) -> None:
    entry = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id or os.environ.get("GB_SESSION", ""),
        "api": api_tokens,
        "local": local_tokens,
    }
    if tok_s > 0:
        entry["tok_s"] = round(tok_s, 1)   # gb-synapse decode speed for this turn's answer
    data = _load(project_dir)
    data["sessions"].append(entry)
    data["totals"]["api"] = data["totals"].get("api", 0) + api_tokens
    data["totals"]["local"] = data["totals"].get("local", 0) + local_tokens
    data["sessions"] = data["sessions"][-100:]
    _save(project_dir, data)


def get_totals(project_dir: Path) -> dict:
    data = _load(project_dir)
    totals = data.get("totals", {"api": 0, "local": 0})

    today = datetime.now().date().isoformat()
    today_api = sum(
        s["api"] for s in data.get("sessions", [])
        if s.get("date", "")[:10] == today
    )
    today_local = sum(
        s["local"] for s in data.get("sessions", [])
        if s.get("date", "")[:10] == today
    )
    return {
        "total_api": totals.get("api", 0),
        "total_local": totals.get("local", 0),
        "today_api": today_api,
        "today_local": today_local,
    }


def format_header_line(project_dir: Path) -> str:
    """Return compact token summary for launcher header display."""
    t = get_totals(project_dir)
    api_today = t["today_api"]
    local_today = t["today_local"]

    def fmt(n: int) -> str:
        return f"{n // 1000}k" if n >= 1000 else str(n)

    if api_today == 0 and local_today == 0:
        return ""
    return f"today: API {fmt(api_today)}  local {fmt(local_today)}"
