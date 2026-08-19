"""
GreenBoost AI Factory — multi-agent autonomous orchestration.

Design → Generate → Simulate → Evaluate → Patch → Repeat

Features:
  - Concurrent agent execution with GreenBoost VRAM-aware scheduling
  - Priority task queue (persisted across restarts)
  - Shared RAG memory across all agents
  - Self-healing: stalled tasks, GPU overload, delegation depth guards
  - Predictive scheduling from task execution history
  - WebSocket broadcast hooks for dashboard integration
  - Strict localhost-only operation (no external network access)

Usage:
  from greenboost_cli.workflow.factory import AIFactory, Task
  factory = AIFactory()
  factory.submit("write unit tests for auth module", priority=5)
  factory.start(workers=2)

  # Scope a task to a project directory and gate it on a real build/test
  # command (the "Evaluate" stage — a turn completing without raising is not
  # proof the code it wrote builds). Gate failures feed back into the retry
  # prompt instead of blindly re-running the same instructions.
  factory.submit(
      "implement the moon-phase calendar UI",
      priority=5,
      metadata={
          "cwd": "/path/to/project",
          "gate_cmd": [["npm", "run", "build"]],
      },
  )
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import sqlite3
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    # Catch broadly: some envs have torch installed but its sub-module
    # imports (e.g. torch.linalg) raise AttributeError. Better to run in
    # no-GPU mode than to break the entire factory plane on a fragile install.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from greenboost_cli.environment.settings import GB_HOME

logger = logging.getLogger("gb.factory")

FACTORY_DB   = GB_HOME / "factory.db"
FACTORY_LOG  = GB_HOME / "factory.log"
MAX_DELEGATION_DEPTH = 5
STALL_TIMEOUT_S      = 300   # 5 min without progress
GPU_WARN_THRESHOLD   = 0.90  # 90% VRAM used
MIN_PRIORITY         = 1     # tasks cannot be bumped below this
MAX_TASK_RETRIES     = 3     # abandon task after this many consecutive failures

# How long a queued-but-never-run task stays resumable across process restarts.
# Rows are only deleted after _run_task() returns, so an interrupted factory
# leaves its work behind; without a bound, that work re-runs in whatever project
# happens to start the next factory, using its own original metadata["cwd"].
# 2h is long enough to resume a genuinely interrupted session and short enough
# that yesterday's abandoned task never ambushes today's.
PENDING_TASK_TTL_S   = float(os.environ.get("GB_FACTORY_PENDING_TTL_S", 2 * 3600))
_SKILL_INJECT_BUDGET_CHARS = 1500  # total skill-body chars injected per task

# Sleep-loop tuning. Hard caps so an autonomous agent cannot exhaust the user's
# API budget while they are AFK. Defaults are deliberately conservative.
SLEEP_LOOP_INTERVAL_S       = 30      # seconds between autonomous task plans
SLEEP_LOOP_MAX_TASKS_HOUR   = 12      # hard cap per agent per hour
SLEEP_LOOP_MAX_PARSE_LINES  = 8       # at most N tasks parsed from one plan
SLEEP_LOOP_DEFAULT_PRIORITY = 15      # autonomous tasks sit below manual ones

# ── gb-synapse HTTP helper (brief expansion + visual QC) ────────────────────
# Same base URL convention as inference/registry.py's "gb-synapse" entry,
# reused here rather than reinvented so both paths track a port override
# (GB_SYNAPSE_PORT) identically.
_GB_SYNAPSE_PORT = int(os.environ.get("GB_SYNAPSE_PORT", "11369"))
_GB_SYNAPSE_URL  = f"http://localhost:{_GB_SYNAPSE_PORT}/v1/chat/completions"

def _gb_synapse_chat(
    messages: list[dict], model: str = "", max_tokens: int = 500, timeout: int = 180,
) -> str:
    """POST to gb-synapse's OpenAI-compatible endpoint and return the reply
    text. `model` is advisory (gb-synapse serves one model at a time and
    ignores a mismatched name rather than erroring, pass the model you
    already confirmed is being served via synapse_serve/synapse_ps). Raises
    RuntimeError with the response body on a non-2xx or malformed reply, so
    callers (both new pipeline stages below) can fold it into their normal
    failure/retry path instead of a raw urllib traceback.

    THINKING-MODEL WARNING (confirmed live, 2026-07-28, building this exact
    pipeline for gb_lunar_calendar): a reasoning-tuned model (e.g. the
    Fable-Fusion default coding model) can burn the entire max_tokens budget
    on its `reasoning_content` before ever writing `content`, returning
    empty text with finish_reason="length", a "/no_think" suffix on the
    user message did NOT suppress this on that model. For prompt-expansion
    (short, low-reasoning-value output), prefer serving a plain instruct
    model for this call (e.g. Qwen3-VL-8B-Instruct-FP8's text path) over a
    thinking model, rather than just raising max_tokens further."""
    import urllib.request

    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(
        _GB_SYNAPSE_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:
        raise RuntimeError(f"gb-synapse request failed: {e}") from e

    try:
        msg = body["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"gb-synapse returned an unexpected shape: {body}") from e

    content = (msg.get("content") or "").strip()
    if not content:
        reason = body["choices"][0].get("finish_reason", "?")
        reasoning_len = len(msg.get("reasoning_content") or "")
        raise RuntimeError(
            f"gb-synapse returned empty content (finish_reason={reason}, "
            f"reasoning_content={reasoning_len} chars), likely a thinking "
            f"model that exhausted max_tokens={max_tokens} on reasoning "
            f"before writing an answer; see _gb_synapse_chat's docstring."
        )
    return content

# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass(order=True)
class Task:
    priority: int
    created_at: float = field(default_factory=time.time, compare=False)
    task_id: str      = field(default="", compare=False)
    prompt: str       = field(default="", compare=False)
    agent_name: str   = field(default="", compare=False)
    autonomous: bool  = field(default=True, compare=False)
    delegated_by: str = field(default="", compare=False)
    delegation_depth: int = field(default=0, compare=False)
    metadata: dict    = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = hashlib.sha256(
                f"{self.prompt}{self.created_at}".encode()
            ).hexdigest()[:12]

    # cwd/gate_cmd/gate_cwd ride in `metadata` (an existing free-form JSON
    # column in FactoryDB) rather than as new dataclass fields, so no DB
    # schema migration is needed and old persisted rows still load fine.

    @property
    def cwd(self) -> str:
        """Absolute path the task's agent should treat as its project root."""
        return self.metadata.get("cwd", "")

    @property
    def gate_cmd(self) -> list[list[str]]:
        """Build/test commands (argv lists) that must pass for the task to
        count as successful. Empty = no gate, exception-only success check."""
        return self.metadata.get("gate_cmd", [])

    @property
    def gate_cwd(self) -> str:
        return self.metadata.get("gate_cwd", "") or self.cwd

    # brief/expand_model/visual_check ride in `metadata` for the same reason
    # cwd/gate_cmd do (see comment above), added 2026-07-28 for the
    # brief-expansion + vision-QC pipeline, gb-app-builder skill.

    @property
    def brief(self) -> str:
        """A short, human-written goal (NOT a precise implementation prompt)
        for _expand_brief() to turn into one before the agent ever sees it.
        Empty means `prompt` is already the real instruction, expansion is
        opt-in per task, existing callers are unaffected."""
        return self.metadata.get("brief", "")

    @property
    def expand_model(self) -> str:
        """Model to expand `brief` with, should be a plain instruct model,
        not a thinking one (see _gb_synapse_chat's docstring). Empty = use
        whatever gb-synapse currently has served."""
        return self.metadata.get("expand_model", "")

    @property
    def expand_context_files(self) -> list[str]:
        """Paths (relative to `cwd`) whose current content gets included as
        context for the brief → prompt expansion call, e.g. the one file
        the resulting task is expected to edit, so the expanded prompt can
        reference real symbol/class names instead of guessing them."""
        return self.metadata.get("expand_context_files", [])

    @property
    def visual_check(self) -> dict:
        """{"url": "http://...", "must_show": ["optional", "keywords"]}.
        When set, _run_visual_check() captures a screenshot of `url` after
        the build gate passes and asks a vision-capable gb-synapse model to
        judge it. Empty dict = no visual check for this task."""
        return self.metadata.get("visual_check", {})


class FactoryGateFailure(Exception):
    """Raised when a task's agent turn completed but its build/test gate failed."""


@dataclass
class AgentStatus:
    name: str
    model: str
    gpu_id: int = 0
    current_task: str = "idle"
    current_task_id: str = ""
    progress: int = 0
    delegated_to: list[str] = field(default_factory=list)
    task_start: float = field(default_factory=time.time)
    paused: bool = False
    total_tasks: int = 0
    failed_tasks: int = 0
    # Skill stack — names from gb's skill registry. Hot-swappable.
    skills: list[str] = field(default_factory=list)
    # Per-agent autonomous loop bookkeeping. Filled by AIFactory only.
    sleep_enabled: bool = False
    # Timestamps of autonomous tasks submitted by this agent (sliding 1h window).
    auto_submit_times: list[float] = field(default_factory=list)
    # Queued skill swap to apply once the current task finishes.
    # tuple: (old_skill_name, new_skill_name)
    pending_swap: Optional[tuple] = None
    # Per-agent lock; used by hot_swap_skill so swaps don't race with task exec.
    # NB: dataclasses can't default to a mutable lock cleanly — we use field
    # with default_factory.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


# ── Task History (for predictive scheduling) ──────────────────────────────────

class TaskHistory:
    """Sliding window of task execution metrics per (agent, task_type)."""

    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._data: dict[tuple[str, str], list[dict]] = {}

    def record(
        self,
        agent: str,
        task_type: str,
        duration: float,
        vram_mb: int,
        success: bool,
    ) -> None:
        key = (agent, task_type)
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(
            {"duration": duration, "vram_mb": vram_mb, "success": success, "ts": time.time()}
        )
        if len(self._data[key]) > self._window:
            self._data[key].pop(0)

    def predict_duration(self, agent: str, task_type: str) -> float:
        hist = self._data.get((agent, task_type), [])
        if not hist:
            return 60.0  # default unknown
        return sum(h["duration"] for h in hist) / len(hist)

    def predict_vram(self, agent: str, task_type: str) -> int:
        hist = self._data.get((agent, task_type), [])
        if not hist:
            return 500  # default unknown MB
        return int(sum(h["vram_mb"] for h in hist) / len(hist))

    def success_rate(self, agent: str, task_type: str) -> float:
        hist = self._data.get((agent, task_type), [])
        if not hist:
            return 0.8  # optimistic default
        return sum(1 for h in hist if h["success"]) / len(hist)


# ── GPU VRAM Monitor ──────────────────────────────────────────────────────────

class VRAMMonitor:
    """GreenBoost-aware VRAM monitoring. Falls back gracefully if unavailable."""

    def free_mb(self, gpu_id: int = 0) -> int:
        try:
            from greenboost_cli.greenboost.monitor import get_tier_stats
            stats = get_tier_stats()
            t1 = stats.get("T1", {})
            return int(t1.get("free_mb", 0))
        except Exception:
            pass
        if TORCH_AVAILABLE:
            try:
                torch.cuda.set_device(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory // (1024 * 1024)
                used  = torch.cuda.memory_allocated(gpu_id) // (1024 * 1024)
                return total - used
            except Exception:
                pass
        return 999_999  # no GPU — unlimited

    def used_mb(self, gpu_id: int = 0) -> int:
        if TORCH_AVAILABLE:
            try:
                torch.cuda.set_device(gpu_id)
                return torch.cuda.memory_allocated(gpu_id) // (1024 * 1024)
            except Exception:
                pass
        return 0

    def usage_ratio(self, gpu_id: int = 0) -> float:
        if TORCH_AVAILABLE:
            try:
                torch.cuda.set_device(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                used  = torch.cuda.memory_allocated(gpu_id)
                return used / max(1, total)
            except Exception:
                pass
        return 0.0


# ── Persistence ───────────────────────────────────────────────────────────────

class FactoryDB:
    """Persists tasks and run history to SQLite so the factory survives restarts."""

    def __init__(self, path: Path = FACTORY_DB) -> None:
        self.path = path
        self._init()

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS pending_tasks (
                task_id         TEXT PRIMARY KEY,
                priority        INTEGER NOT NULL,
                created_at      REAL NOT NULL,
                prompt          TEXT NOT NULL,
                agent_name      TEXT NOT NULL DEFAULT '',
                autonomous      INTEGER NOT NULL DEFAULT 1,
                delegated_by    TEXT NOT NULL DEFAULT '',
                delegation_depth INTEGER NOT NULL DEFAULT 0,
                metadata        TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS completed_tasks (
                task_id     TEXT PRIMARY KEY,
                prompt      TEXT,
                agent_name  TEXT,
                started_at  REAL,
                finished_at REAL,
                success     INTEGER,
                output_len  INTEGER,
                error       TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_task_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                agent       TEXT NOT NULL,
                task_type   TEXT NOT NULL,
                duration    REAL NOT NULL,
                vram_mb     INTEGER NOT NULL DEFAULT 0,
                success     INTEGER NOT NULL DEFAULT 1,
                ts          REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_state (
                name            TEXT PRIMARY KEY,
                model           TEXT NOT NULL,
                gpu_id          INTEGER NOT NULL DEFAULT 0,
                skills          TEXT NOT NULL DEFAULT '[]',
                sleep_enabled   INTEGER NOT NULL DEFAULT 0,
                paused          INTEGER NOT NULL DEFAULT 0,
                updated_at      REAL NOT NULL
            );
            """)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def save_task(self, task: Task) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO pending_tasks
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    task.task_id, task.priority, task.created_at,
                    task.prompt, task.agent_name, int(task.autonomous),
                    task.delegated_by, task.delegation_depth,
                    json.dumps(task.metadata),
                ),
            )

    def delete_task(self, task_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM pending_tasks WHERE task_id=?", (task_id,))

    def load_pending(self, ttl_s: float = 0.0) -> list[Task]:
        """Restore queued tasks, dropping any older than `ttl_s` (0 = keep all).

        `delete_task()` only runs AFTER `_run_task()` returns, so a factory whose
        process exits mid-task — an interrupted run, a killed driver, a crash —
        leaves its row behind. The next `AIFactory()` in any project then
        re-queues it unconditionally, carrying its ORIGINAL `metadata["cwd"]`,
        and starts editing files in a directory the current session never
        mentioned.

        Observed 2026-08-17: a completed 'edit src/game.ts' task from one
        project was resurrected 37 minutes later at the start of an unrelated
        project's first run, jumped ahead of that session's own task on
        `created_at ASC`, and took the single local model with it. Nothing was
        corrupted (the stale cwd pointed at its own project) but the new run was
        blocked behind work nobody asked for.

        Expired rows are deleted and returned to the caller so the drop is
        logged rather than silent — a task vanishing without explanation is its
        own debugging problem.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM pending_tasks ORDER BY priority ASC, created_at ASC"
            ).fetchall()
        if ttl_s and ttl_s > 0:
            cutoff = time.time() - ttl_s
            fresh, stale = [], []
            for r in rows:
                (stale if (r["created_at"] or 0) < cutoff else fresh).append(r)
            if stale:
                with self._conn() as c:
                    for r in stale:
                        c.execute("DELETE FROM pending_tasks WHERE task_id=?",
                                  (r["task_id"],))
                self.expired_on_load = [
                    {"task_id": r["task_id"],
                     "age_s": round(time.time() - (r["created_at"] or 0)),
                     "prompt": (r["prompt"] or "")[:80]}
                    for r in stale
                ]
            rows = fresh
        return [
            Task(
                priority=r["priority"],
                created_at=r["created_at"],
                task_id=r["task_id"],
                prompt=r["prompt"],
                agent_name=r["agent_name"],
                autonomous=bool(r["autonomous"]),
                delegated_by=r["delegated_by"],
                delegation_depth=r["delegation_depth"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    def complete_task(
        self,
        task_id: str,
        agent: str,
        started_at: float,
        success: bool,
        output_len: int = 0,
        error: str = "",
    ) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO completed_tasks
                   VALUES (?,?,?,?,?,?,?,?)""",
                (task_id, "", agent, started_at, time.time(),
                 int(success), output_len, error),
            )

    def save_history(
        self, agent: str, task_type: str, duration: float, vram_mb: int, success: bool
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO agent_task_history (agent,task_type,duration,vram_mb,success,ts) "
                "VALUES (?,?,?,?,?,?)",
                (agent, task_type, duration, vram_mb, int(success), time.time()),
            )

    def recent_completions(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM completed_tasks ORDER BY finished_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Agent state persistence ──────────────────────────────────────────────

    def save_agent_state(
        self,
        name: str,
        model: str,
        gpu_id: int,
        skills: list[str],
        sleep_enabled: bool,
        paused: bool,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO agent_state
                   (name, model, gpu_id, skills, sleep_enabled, paused, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (name, model, int(gpu_id), json.dumps(list(skills)),
                 int(bool(sleep_enabled)), int(bool(paused)), time.time()),
            )

    def delete_agent_state(self, name: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM agent_state WHERE name=?", (name,))

    def load_agent_states(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM agent_state").fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                skills = json.loads(r["skills"] or "[]")
                if not isinstance(skills, list):
                    skills = []
            except Exception:
                skills = []
            out.append({
                "name":          r["name"],
                "model":         r["model"],
                "gpu_id":        r["gpu_id"],
                "skills":        skills,
                "sleep_enabled": bool(r["sleep_enabled"]),
                "paused":        bool(r["paused"]),
            })
        return out


# ── Broadcast Bus ─────────────────────────────────────────────────────────────
# Decoupled from asyncio — dashboard polls via REST or SSE, not WebSocket.
# This keeps the factory purely threaded (no asyncio dependency).

_broadcast_callbacks: list[Callable[[dict], None]] = []


def register_broadcast(cb: Callable[[dict], None]) -> None:
    """Dashboard registers a callback here; factory calls it for state changes."""
    _broadcast_callbacks.append(cb)


def _emit(event: str, data: dict) -> None:
    msg = {"event": event, "ts": time.time(), **data}
    for cb in _broadcast_callbacks:
        try:
            cb(msg)
        except Exception:
            pass


# ── AI Factory ────────────────────────────────────────────────────────────────

class AIFactory:
    """
    Multi-agent task execution engine with GreenBoost VRAM-aware scheduling.

    Security notes:
      - Tasks are never executed with shell=True unless 'shell_exec' in metadata
        and user has explicitly enabled autonomous_shell in settings.
      - Delegation depth is capped at MAX_DELEGATION_DEPTH to prevent loops.
      - All inter-agent communication is in-process — no external network calls.
    """

    def __init__(
        self,
        max_depth: int = MAX_DELEGATION_DEPTH,
        stall_timeout: float = STALL_TIMEOUT_S,
        gpu_warn_threshold: float = GPU_WARN_THRESHOLD,
        min_priority: int = MIN_PRIORITY,
        max_retries: int = MAX_TASK_RETRIES,
    ) -> None:
        self.max_depth           = max_depth
        self.stall_timeout       = stall_timeout
        self.gpu_warn_threshold  = gpu_warn_threshold
        self.min_priority        = min_priority
        self.max_retries         = max_retries
        self.db         = FactoryDB()
        self.vram       = VRAMMonitor()
        self.history    = TaskHistory()
        self._task_q: queue.PriorityQueue[Task] = queue.PriorityQueue()
        self._agents: dict[str, AgentStatus] = {}
        self._lock      = threading.Lock()
        self._active    = False
        self._workers: list[threading.Thread] = []
        self._sleep_threads: dict[str, threading.Thread] = {}
        self._sleep_enabled_default = False
        self._log       = self._setup_log()

        # Reload persisted pending tasks, minus anything stale enough that
        # re-running it now would be a surprise rather than a resumption
        # (see TaskDB.load_pending's docstring for the incident).
        self.db.expired_on_load = []
        for task in self.db.load_pending(ttl_s=PENDING_TASK_TTL_S):
            self._task_q.put(task)
        for exp in getattr(self.db, "expired_on_load", []):
            self._log.warning(
                "dropped stale pending task %s (age %ss): %s",
                exp["task_id"][:12], exp["age_s"], exp["prompt"],
            )
            _emit("factory_task_expired", exp)

        # Reload persisted agent state (skills, sleep flag, paused).
        for st in self.db.load_agent_states():
            self._agents[st["name"]] = AgentStatus(
                name=st["name"],
                model=st["model"],
                gpu_id=st["gpu_id"],
                skills=list(st["skills"]),
                sleep_enabled=bool(st["sleep_enabled"]),
                paused=bool(st["paused"]),
            )

    def _setup_log(self) -> logging.Logger:
        FACTORY_LOG.parent.mkdir(parents=True, exist_ok=True)
        log = logging.getLogger("gb.factory")
        if not log.handlers:
            fh = logging.FileHandler(FACTORY_LOG)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(fh)
            log.setLevel(logging.INFO)
        return log

    # ── Agent management ──────────────────────────────────────────────────────

    def add_agent(
        self,
        name: str,
        model: str = "claude-sonnet-4-6",
        gpu_id: int = 0,
        skills: Optional[list[str]] = None,
    ) -> None:
        with self._lock:
            if name in self._agents:
                # Preserve existing state if re-adding; only refresh model/gpu/skills.
                a = self._agents[name]
                a.model = model
                a.gpu_id = gpu_id
                if skills is not None:
                    a.skills = list(skills)
            else:
                self._agents[name] = AgentStatus(
                    name=name, model=model, gpu_id=gpu_id,
                    skills=list(skills or []),
                )
            self._persist_agent_locked(name)
        # Auto-start a sleep thread if sleep mode is default-on for new agents.
        if self._sleep_enabled_default and self._active:
            self._ensure_sleep_thread(name)
        _emit("agent_added", {"agent": name, "model": model})
        self._log.info("Agent added: %s (%s)", name, model)

    def remove_agent(self, name: str) -> None:
        with self._lock:
            self._agents.pop(name, None)
        self.db.delete_agent_state(name)
        # The sleep thread (if any) checks _agents on each iteration and exits.
        self._sleep_threads.pop(name, None)
        _emit("agent_removed", {"agent": name})

    def pause_agent(self, name: str) -> None:
        with self._lock:
            if name in self._agents:
                self._agents[name].paused = True
                self._persist_agent_locked(name)
        _emit("agent_paused", {"agent": name})

    def resume_agent(self, name: str) -> None:
        with self._lock:
            if name in self._agents:
                self._agents[name].paused = False
                self._persist_agent_locked(name)
        _emit("agent_resumed", {"agent": name})

    def list_agents(self) -> list[dict]:
        """Detailed per-agent listing (used by the headless CLI)."""
        now = time.time()
        with self._lock:
            out = []
            for a in self._agents.values():
                idle = now - a.task_start if a.current_task == "idle" else 0.0
                out.append({
                    "name":           a.name,
                    "model":          a.model,
                    "gpu_id":         a.gpu_id,
                    "current_task":   a.current_task,
                    "current_task_id": a.current_task_id,
                    "progress":       a.progress,
                    "paused":         a.paused,
                    "skills":         list(a.skills),
                    "sleep_enabled":  a.sleep_enabled,
                    "total_tasks":    a.total_tasks,
                    "failed_tasks":   a.failed_tasks,
                    "idle_seconds":   round(idle, 1),
                    "vram_used_mb":   self.vram.used_mb(a.gpu_id),
                    "vram_free_mb":   self.vram.free_mb(a.gpu_id),
                    "pending_swap":   list(a.pending_swap) if a.pending_swap else None,
                })
        return out

    def _persist_agent_locked(self, name: str) -> None:
        """Caller must hold self._lock."""
        a = self._agents.get(name)
        if not a:
            return
        try:
            self.db.save_agent_state(
                a.name, a.model, a.gpu_id, list(a.skills),
                a.sleep_enabled, a.paused,
            )
        except Exception as e:
            self._log.warning("Failed to persist agent state for %s: %s", name, e)

    # ── Skill hot-swap ────────────────────────────────────────────────────────

    def hot_swap_skill(
        self,
        agent_name: str,
        old_skill_name: str,
        new_skill_name: str,
    ) -> dict:
        """Replace `old_skill_name` with `new_skill_name` in an agent's skill stack.

        Atomic w.r.t. the agent: uses the agent's per-instance lock so no swap
        races with a concurrent task. If the agent is currently executing a
        task, the swap is queued and applied immediately after the task
        finishes (see _apply_pending_swap_locked).

        Returns a dict describing the outcome (status: "applied"|"queued"|"error").
        """
        with self._lock:
            agent = self._agents.get(agent_name)
            if agent is None:
                return {"status": "error", "error": f"agent '{agent_name}' not found"}
        # Take the agent-level lock outside the factory lock to avoid deadlock
        # with the worker (worker only grabs factory lock).
        with agent.lock:
            if agent.current_task != "idle":
                agent.pending_swap = (old_skill_name, new_skill_name)
                _emit("skill_swap_queued", {
                    "agent": agent_name,
                    "old": old_skill_name,
                    "new": new_skill_name,
                })
                self._log.info(
                    "Hot-swap queued for %s: %s -> %s (busy)",
                    agent_name, old_skill_name, new_skill_name,
                )
                return {
                    "status": "queued",
                    "agent": agent_name,
                    "old": old_skill_name,
                    "new": new_skill_name,
                }

            ok = self._apply_swap_locked(agent, old_skill_name, new_skill_name)
            if not ok:
                return {
                    "status": "error",
                    "error": f"skill '{old_skill_name}' not in agent's stack",
                    "agent": agent_name,
                    "skills": list(agent.skills),
                }

        # Persist outside the agent lock; reacquire factory lock briefly.
        with self._lock:
            self._persist_agent_locked(agent_name)
        _emit("skill_swap_applied", {
            "agent": agent_name,
            "old": old_skill_name,
            "new": new_skill_name,
            "skills": list(agent.skills),
        })
        self._log.info(
            "Hot-swap applied for %s: %s -> %s", agent_name, old_skill_name, new_skill_name,
        )
        return {
            "status": "applied",
            "agent": agent_name,
            "old": old_skill_name,
            "new": new_skill_name,
            "skills": list(agent.skills),
        }

    def _apply_swap_locked(
        self, agent: AgentStatus, old: str, new: str,
    ) -> bool:
        """Caller must hold agent.lock."""
        if old not in agent.skills:
            return False
        idx = agent.skills.index(old)
        if new and new not in agent.skills:
            agent.skills[idx] = new
        else:
            # If `new` is empty or already present, drop the old one.
            agent.skills.pop(idx)
        agent.pending_swap = None
        return True

    def _apply_pending_swap(self, agent: AgentStatus) -> None:
        """Called by the worker after a task completes."""
        with agent.lock:
            if not agent.pending_swap:
                return
            old, new = agent.pending_swap
            applied = self._apply_swap_locked(agent, old, new)
        if applied:
            with self._lock:
                self._persist_agent_locked(agent.name)
            _emit("skill_swap_applied", {
                "agent": agent.name, "old": old, "new": new,
                "skills": list(agent.skills),
            })
            self._log.info(
                "Deferred hot-swap applied for %s: %s -> %s", agent.name, old, new,
            )

    # ── Task submission ───────────────────────────────────────────────────────

    def submit(
        self,
        prompt: str = "",
        agent_name: str = "",
        priority: int = 10,
        autonomous: bool = True,
        delegated_by: str = "",
        delegation_depth: int = 0,
        metadata: dict | None = None,
        brief: str = "",
        expand_model: str = "",
        expand_context_files: list[str] | None = None,
        visual_check_url: str = "",
        visual_check_must_show: list[str] | None = None,
        visual_check_model: str = "",
    ) -> str:
        """`prompt` OR `brief`, not both, `brief` is a short goal that
        _expand_brief() turns into the real prompt at run time (see Task.brief
        and _gb_synapse_chat's docstring re: picking a non-thinking
        expand_model). `visual_check_url` opts the task into a post-gate
        screenshot + vision-model judgment (Task.visual_check) instead of
        (or in addition to) a text-only gate_cmd in `metadata`. Both are
        additive convenience kwargs over `metadata`, equivalent to setting
        metadata["brief"]/["visual_check"] by hand."""
        priority = max(self.min_priority, int(priority))
        if delegation_depth > self.max_depth:
            self._log.warning("Delegation depth exceeded for: %s", (prompt or brief)[:60])
            _emit("alert", {"level": "warning", "msg": f"Max delegation depth reached: {(prompt or brief)[:40]}"})
            return ""

        full_metadata = dict(metadata or {})
        if brief:
            full_metadata["brief"] = brief
            if expand_model:
                full_metadata["expand_model"] = expand_model
            if expand_context_files:
                full_metadata["expand_context_files"] = expand_context_files
        if visual_check_url:
            full_metadata["visual_check"] = {
                "url": visual_check_url,
                "must_show": visual_check_must_show or [],
                "model": visual_check_model,
            }

        task = Task(
            priority=priority,
            prompt=prompt,
            agent_name=agent_name,
            autonomous=autonomous,
            delegated_by=delegated_by,
            delegation_depth=delegation_depth,
            metadata=full_metadata,
        )
        self._task_q.put(task)
        self.db.save_task(task)
        _emit("task_submitted", {"task_id": task.task_id, "prompt": (prompt or brief)[:80]})
        return task.task_id

    # ── VRAM-aware agent selection ────────────────────────────────────────────

    def _select_agent(self, task: Task) -> Optional[str]:
        """Pick the best available agent for a task using predictive history."""
        with self._lock:
            candidates = [
                a for a in self._agents.values()
                if not a.paused and a.current_task == "idle"
            ]
        if not candidates:
            return None

        task_words = (task.prompt or task.brief).split()
        task_type = task_words[0].lower() if task_words else "unknown"

        best: Optional[str] = None
        best_score = float("inf")

        for agent in candidates:
            # Check VRAM headroom
            predicted_vram = self.history.predict_vram(agent.name, task_type)
            free_mb        = self.vram.free_mb(agent.gpu_id)
            if free_mb < predicted_vram:
                continue  # not enough memory

            # Score = predicted_duration / success_rate (lower = better)
            dur    = self.history.predict_duration(agent.name, task_type)
            sr     = self.history.success_rate(agent.name, task_type)
            score  = dur / max(sr, 0.1)
            if score < best_score:
                best_score = score
                best = agent.name

        # Fall back to any idle agent if no history
        if best is None and candidates:
            best = candidates[0].name
        return best

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while self._active:
            try:
                task = self._task_q.get(timeout=2.0)
            except queue.Empty:
                self._health_check()
                continue

            # Check if GPU is overloaded — requeue if so
            usage = self.vram.usage_ratio()
            if usage > self.gpu_warn_threshold:
                self._log.warning("GPU overload (%.0f%%) — requeueing task", usage * 100)
                _emit("alert", {"level": "warning", "msg": f"GPU {usage*100:.0f}% — task requeued"})
                self._task_q.put(Task(priority=max(self.min_priority, task.priority - 1), **{
                    k: getattr(task, k) for k in (
                        "created_at", "task_id", "prompt", "agent_name",
                        "autonomous", "delegated_by", "delegation_depth", "metadata"
                    )
                }))
                time.sleep(1)
                continue

            agent_name = (
                task.agent_name
                or self._select_agent(task)
                or (list(self._agents.keys())[0] if self._agents else None)
            )
            if not agent_name or agent_name not in self._agents:
                self._log.error("No agent available for task %s", task.task_id)
                self._task_q.task_done()
                continue

            self._run_task(agent_name, task)
            self._task_q.task_done()
            self.db.delete_task(task.task_id)

    def _run_task(self, agent_name: str, task: Task) -> None:
        agent = self._agents.get(agent_name)
        if agent is None or agent.paused:
            self._task_q.put(task)
            return

        start    = time.time()
        vram_before = self.vram.used_mb(agent.gpu_id)

        display_prompt = task.prompt or task.brief

        with self._lock:
            agent.current_task    = display_prompt[:60]
            agent.current_task_id = task.task_id
            agent.task_start      = start
            agent.progress        = 0

        _emit("task_started", {
            "agent": agent_name, "task_id": task.task_id,
            "prompt": display_prompt[:80],
        })

        try:
            if task.brief and not task.prompt:
                task.prompt = self._expand_brief(task)
            output = self._invoke_agent(agent, task)
            if task.gate_cmd:
                gate_ok, gate_report = self._run_gate(task)
                if not gate_ok:
                    raise FactoryGateFailure(gate_report)
            if task.visual_check:
                visual_ok, visual_report = self._run_visual_check(task)
                if not visual_ok:
                    raise FactoryGateFailure(visual_report)
            success = True
            with self._lock:
                agent.total_tasks += 1
        except Exception as exc:
            output = str(exc)[:1000]
            success = False
            with self._lock:
                agent.failed_tasks += 1
            self._log.exception("Task %s failed on agent %s", task.task_id, agent_name)
            _emit("alert", {"level": "error", "agent": agent_name,
                             "msg": f"Task failed: {exc!s:.100}"})

            retries = task.metadata.get("_retries", 0)
            if retries < self.max_retries:
                retry_prompt = task.prompt
                if isinstance(exc, FactoryGateFailure):
                    retry_prompt = (
                        f"{task.prompt}\n\n"
                        f"Your previous attempt did not pass its build/test gate. "
                        f"Fix the following errors in the files you already "
                        f"created/edited, do not start over:\n\n{str(exc)[:1500]}"
                    )
                retry_task = Task(
                    priority=task.priority,
                    task_id=task.task_id,
                    prompt=retry_prompt,
                    agent_name=task.agent_name,
                    autonomous=task.autonomous,
                    delegated_by=task.delegated_by,
                    delegation_depth=task.delegation_depth,
                    metadata={**task.metadata, "_retries": retries + 1},
                )
                self._task_q.put(retry_task)
                self.db.save_task(retry_task)
                self._log.warning(
                    "Task %s retry %d/%d", task.task_id, retries + 1, self.max_retries
                )
            else:
                self._log.error(
                    "Task %s exhausted %d retries — dropped", task.task_id, self.max_retries
                )
                _emit("alert", {"level": "error", "agent": agent_name,
                                 "msg": f"Task abandoned after {self.max_retries} retries: {task.prompt[:60]}"})

        duration  = time.time() - start
        vram_used = max(0, self.vram.used_mb(agent.gpu_id) - vram_before)
        task_words = (task.prompt or task.brief).split()
        task_type = task_words[0].lower() if task_words else "unknown"

        self.history.record(agent_name, task_type, duration, vram_used, success)
        self.db.save_history(agent_name, task_type, duration, vram_used, success)
        self.db.complete_task(
            task.task_id, agent_name, start, success,
            len(output) if output else 0,
            output if not success else "",
        )

        with self._lock:
            agent.current_task    = "idle"
            agent.current_task_id = ""
            agent.progress        = 100

        # Apply any skill swap that was queued while the task was running.
        self._apply_pending_swap(agent)

        _emit("task_completed", {
            "agent": agent_name, "task_id": task.task_id,
            "success": success, "duration": round(duration, 1),
        })
        self._log.info(
            "Task %s done in %.1fs success=%s agent=%s",
            task.task_id, duration, success, agent_name,
        )

    def _invoke_agent(self, agent: AgentStatus, task: Task) -> str:
        """Invoke the agent's backend model for the task prompt."""
        try:
            from greenboost_cli.core.orchestrator import execute_turn_sync
            from greenboost_cli.core.session import ConversationSession
            from greenboost_cli.environment.settings import load_settings
            _settings = load_settings()
            _settings["model"] = agent.model
            _settings["permission_mode"] = "accept-all"
            _settings["_agent_label"] = task.task_id

            system_context = ""
            if agent.skills:
                # agent.skills was stored on AgentStatus but never actually
                # read anywhere in the real task-execution path — assigning a
                # skill to an agent was a silent no-op. Loaded here, budget-
                # capped: factory tasks already carry a cwd directive and
                # (on retry) accumulated gate-failure text, and a combined
                # prompt that's too large for this model's ctx budget fails
                # outright rather than degrading gracefully (confirmed live,
                # 2026-07-27: a 400 "request exceeds the available context
                # size" error) — skill injection must stay small.
                system_context += self._skills_context(agent.skills)
            if task.cwd:
                from greenboost_cli.instruments.handlers import set_task_bash_cwd
                # Thread-local: safe under concurrent factory workers, each
                # running a different task's turn on its own OS thread.
                set_task_bash_cwd(task.cwd)
                system_context += (
                    f"Your working directory for this task is: {task.cwd}\n"
                    f"Use absolute paths under this directory for every file "
                    f"read, write, edit, and shell command. Do not operate "
                    f"outside this directory unless explicitly asked to.\n\n"
                )
                # NemoClaw audit, Phase 3c: the prompt directive above asked
                # nicely; this makes it enforcement. A factory task's Write/
                # Edit calls are jailed to task.cwd via the same tool-policy
                # mechanism Phase 3b wires through dispatch() — Read/Bash
                # stay unrestricted by this (see instruments/policy.py).
                from greenboost_cli.instruments.policy import build_policy
                _settings["_tool_policy"] = build_policy(workspace_roots=[task.cwd])
            return execute_turn_sync(task.prompt, ConversationSession(), _settings, system_context)
        except ImportError:
            # Fallback: return a stub so the factory stays operational
            return f"[stub] Agent {agent.name} processed: {task.prompt[:60]}"

    def _skills_context(self, skill_names: list[str]) -> str:
        """Load an agent's assigned skills' SKILL.md bodies for injection.
        Deliberately small budget — see the comment at the _invoke_agent call
        site for why a factory task's total prompt must stay ctx-frugal."""
        try:
            from greenboost_cli.skill.router import (
                discover_all_skill_dirs, discover_skills_multi, load_skill_body,
            )
            from greenboost_cli.environment.settings import load_settings
            dirs = discover_all_skill_dirs(load_settings())
            entries = {e.name: e for e in discover_skills_multi(dirs)}
            budget = _SKILL_INJECT_BUDGET_CHARS
            chunks: list[str] = []
            for name in skill_names:
                entry = entries.get(name)
                if not entry or budget <= 0:
                    continue
                body = load_skill_body(Path(entry.path), max_chars=min(budget, 1200))
                chunks.append(f"# Skill: {name}\n{body}")
                budget -= len(body)
            return ("\n\n".join(chunks) + "\n\n") if chunks else ""
        except Exception:
            return ""

    def _expand_brief(self, task: Task) -> str:
        """Turn task.brief (a short human goal) into a precise implementation
        prompt for the coding agent, the "prompt generates prompt" step of
        the gb-app-builder pipeline (owner directive 2026-07-28: Fable-Fusion
        should generate the task prompt, not just execute a hand-written
        one). Includes task.expand_context_files' CURRENT content so the
        expansion can name real symbols instead of inventing them, mirroring
        why app_spec.py reads real files rather than working from the brief
        alone. Raises (via _gb_synapse_chat) rather than silently falling
        back to the raw brief, a silent fallback would hide a broken
        expansion step behind a plausible-looking but underspecified prompt,
        exactly the failure mode this whole pipeline exists to avoid."""
        context = ""
        for rel in task.expand_context_files:
            path = Path(task.cwd) / rel if task.cwd else Path(rel)
            try:
                context += f"\n\nCurrent content of {rel}:\n```\n{path.read_text()}\n```"
            except OSError as e:
                context += f"\n\n(could not read {rel}: {e})"

        meta_prompt = (
            "You are writing a precise implementation instruction (a task "
            "prompt) for a coding agent. The agent will act on your "
            "instruction alone, it will not see this message. Be exact "
            "about file paths, class/element names, and what NOT to touch. "
            f"Output ONLY the instruction text, max 150 words.\n\nGoal: "
            f"{task.brief}{context}"
        )
        return _gb_synapse_chat(
            [{"role": "user", "content": meta_prompt}],
            model=task.expand_model,
            max_tokens=int(task.metadata.get("expand_max_tokens", 500)),
        )

    def _run_visual_check(self, task: Task) -> tuple[bool, str]:
        """Screenshot task.visual_check['url'] and ask a vision-capable
        gb-synapse model to judge it, the visual-QA half of "Evaluate" a
        text-only build gate cannot cover (owner directive 2026-07-28: the
        quality check must go through an actual vision model looking at the
        rendered page, not Claude reading JSX). The model MUST answer
        PASS or FAIL: <reason> on its first line, parsed strictly so a
        FAIL becomes a real FactoryGateFailure that feeds the exact same
        retry-with-feedback loop _run_gate's failures already use, not just
        a logged opinion nobody acts on.

        For a Tauri app, only point this at a view that renders without a
        Tauri `invoke()` call, a plain `npm run dev` URL has no Tauri IPC
        bridge, so any view depending on backend data will screenshot as an
        error state, not a real bug, see screenshot_utils.py's docstring."""
        from greenboost_cli.instruments.screenshot_utils import capture_screenshot

        vc = task.visual_check
        shot_path = str(Path(task.cwd or ".") / f".gb_visual_check_{task.task_id}.png")
        capture_result = capture_screenshot(vc["url"], shot_path)
        if capture_result.startswith("Error:"):
            return False, capture_result

        import base64
        b64 = base64.b64encode(Path(shot_path).read_bytes()).decode()
        must_show = vc.get("must_show", [])
        checklist = f"\nIt must visibly show: {', '.join(must_show)}." if must_show else ""
        prompt = (
            "Judge this screenshot of a web/desktop app UI for visual "
            "correctness, layout broken, unstyled/raw elements, missing "
            "content, obviously wrong colors or overlap." + checklist +
            "\nAnswer with PASS on the first line if it looks correct, or "
            "FAIL: <short reason> on the first line if not. Nothing else "
            "on that line."
        )
        try:
            verdict = _gb_synapse_chat(
                [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}],
                model=vc.get("model", ""),
                max_tokens=120,
            )
        except RuntimeError as e:
            return False, f"visual check request failed: {e}"

        first_line = verdict.strip().splitlines()[0] if verdict.strip() else ""
        if first_line.upper().startswith("PASS"):
            return True, verdict
        return False, verdict

    def _run_gate(self, task: Task) -> tuple[bool, str]:
        """Run the task's build/test gate commands. Real "Evaluate" stage:
        an agent turn completing without an exception is not proof the code
        it wrote actually builds — this runs the commands the caller supplied
        and reports pass/fail plus combined output for the retry prompt."""
        gate_cwd = task.gate_cwd or os.getcwd()
        reports: list[str] = []
        ok = True
        for cmd in task.gate_cmd:
            try:
                proc = subprocess.run(
                    cmd, cwd=gate_cwd, capture_output=True, text=True, timeout=300,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                ok = False
                reports.append(f"$ {' '.join(cmd)}\n<gate error: {e}>")
                continue
            reports.append(f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
            if proc.returncode != 0:
                ok = False
        return ok, "\n\n".join(reports)

    # ── Health check (stall detection) ───────────────────────────────────────

    def _health_check(self) -> None:
        now = time.time()
        with self._lock:
            for agent in list(self._agents.values()):
                if agent.current_task == "idle" or agent.paused:
                    continue
                stalled = now - agent.task_start > STALL_TIMEOUT_S
                if stalled:
                    self._log.warning("Agent %s appears stalled on: %s",
                                      agent.name, agent.current_task)
                    _emit("alert", {
                        "level": "warning",
                        "agent": agent.name,
                        "msg": f"Stalled on: {agent.current_task[:50]}",
                    })
                    # Reset so the worker can pick up new tasks
                    agent.current_task = "idle"
                    agent.progress     = 0

    # ── Autonomous "sleep" loop ──────────────────────────────────────────────

    def set_sleep(
        self,
        enabled: bool,
        agent_name: Optional[str] = None,
    ) -> dict:
        """Toggle the autonomous task-generation loop.

        If `agent_name` is provided, only that agent's loop is affected.
        Otherwise, the toggle applies to every registered agent AND becomes the
        default for any agent added later (until set_sleep is called again).
        """
        changed: list[str] = []
        with self._lock:
            if agent_name is None:
                self._sleep_enabled_default = bool(enabled)
                names = list(self._agents.keys())
            else:
                if agent_name not in self._agents:
                    return {"status": "error",
                            "error": f"agent '{agent_name}' not found"}
                names = [agent_name]

            for n in names:
                a = self._agents[n]
                if a.sleep_enabled != bool(enabled):
                    a.sleep_enabled = bool(enabled)
                    self._persist_agent_locked(n)
                    changed.append(n)

        # Start / stop sleep threads outside the lock.
        if enabled and self._active:
            for n in names:
                self._ensure_sleep_thread(n)
        # Threads exit on their own when sleep_enabled flips to False.

        _emit("sleep_toggled", {
            "enabled": bool(enabled),
            "agent": agent_name,
            "changed": changed,
        })
        return {
            "status":  "ok",
            "enabled": bool(enabled),
            "agent":   agent_name,
            "changed": changed,
        }

    def _ensure_sleep_thread(self, agent_name: str) -> None:
        t = self._sleep_threads.get(agent_name)
        if t is not None and t.is_alive():
            return
        nt = threading.Thread(
            target=self._agent_sleep_loop,
            name=f"gb-factory-sleep-{agent_name}",
            args=(agent_name,),
            daemon=True,
        )
        self._sleep_threads[agent_name] = nt
        nt.start()

    def _agent_sleep_loop(self, agent_name: str) -> None:
        """Periodically ask the agent to plan its next autonomous task.

        Heavily rate-limited: at most SLEEP_LOOP_MAX_TASKS_HOUR tasks per agent
        per rolling hour. The loop exits cleanly when:
          - the factory stops (self._active False),
          - the agent is removed,
          - the agent's sleep_enabled is set False.
        """
        self._log.info("Sleep loop started for agent=%s", agent_name)
        while self._active:
            with self._lock:
                agent = self._agents.get(agent_name)
                if agent is None or not agent.sleep_enabled:
                    break
                paused = agent.paused

            if paused:
                time.sleep(SLEEP_LOOP_INTERVAL_S)
                continue

            # Enforce the rolling-hour cap.
            now = time.time()
            with self._lock:
                agent.auto_submit_times = [
                    t for t in agent.auto_submit_times if (now - t) < 3600.0
                ]
                used = len(agent.auto_submit_times)
            if used >= SLEEP_LOOP_MAX_TASKS_HOUR:
                # Sleep until the oldest entry falls outside the window.
                time.sleep(SLEEP_LOOP_INTERVAL_S)
                continue

            try:
                plan = self._plan_autonomous_tasks(agent_name)
            except Exception as e:
                self._log.warning("Sleep loop plan failed for %s: %s", agent_name, e)
                plan = []

            for line in plan[:SLEEP_LOOP_MAX_PARSE_LINES]:
                with self._lock:
                    agent = self._agents.get(agent_name)
                    if agent is None or not agent.sleep_enabled:
                        break
                    agent.auto_submit_times.append(time.time())
                    if len(agent.auto_submit_times) >= SLEEP_LOOP_MAX_TASKS_HOUR:
                        # Don't overshoot the cap within this batch.
                        pass

                self.submit(
                    prompt=line,
                    agent_name=agent_name,
                    priority=SLEEP_LOOP_DEFAULT_PRIORITY,
                    autonomous=True,
                    metadata={"_autonomous": True, "_source": "sleep_loop"},
                )

                with self._lock:
                    agent = self._agents.get(agent_name)
                    if agent is None:
                        break
                    if len(agent.auto_submit_times) >= SLEEP_LOOP_MAX_TASKS_HOUR:
                        break

            time.sleep(SLEEP_LOOP_INTERVAL_S)

        self._log.info("Sleep loop exited for agent=%s", agent_name)

    def _plan_autonomous_tasks(self, agent_name: str) -> list[str]:
        """Ask the agent's backend for the next autonomous tasks.

        Returns a list of prompt strings, one per task. Bounded by
        SLEEP_LOOP_MAX_PARSE_LINES upstream. If the orchestrator backend isn't
        importable or fails, returns []. The factory must keep running.
        """
        with self._lock:
            agent = self._agents.get(agent_name)
            if agent is None:
                return []
            model = agent.model
            skills = list(agent.skills)

        plan_prompt = (
            "You are running in autonomous sleep mode. Based on prior context, "
            "list up to 5 concrete next tasks you should run. One per line. "
            "Do not include numbering, bullets, or commentary — just the task "
            "prompts as plain text lines. Skip if there is nothing useful to do."
        )
        if skills:
            plan_prompt += f"\nAvailable skills: {', '.join(skills)}"

        try:
            from greenboost_cli.core.orchestrator import execute_turn_sync
            from greenboost_cli.core.session import ConversationSession
            from greenboost_cli.environment.settings import load_settings
            _settings = load_settings()
            _settings["model"] = model
            _settings["permission_mode"] = "accept-all"
            raw = execute_turn_sync(plan_prompt, ConversationSession(), _settings, "")
        except ImportError:
            return []
        except Exception as e:
            self._log.warning("Autonomous plan call failed (%s): %s", agent_name, e)
            return []

        lines: list[str] = []
        for raw_line in (raw or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Strip common bullet/number prefixes.
            if line[0] in "-*•":
                line = line[1:].strip()
            while line and line[0].isdigit():
                line = line[1:].lstrip(".) ").strip()
            if 5 <= len(line) <= 500:
                lines.append(line)
            if len(lines) >= SLEEP_LOOP_MAX_PARSE_LINES:
                break
        return lines

    # ── Factory lifecycle ─────────────────────────────────────────────────────

    def start(self, workers: int = 2, sleep: bool = False) -> None:
        if self._active:
            return
        if not self._agents:
            self.add_agent("default", model="claude-sonnet-4-6")

        self._active = True
        for i in range(workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"gb-factory-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

        # Opt-in sleep loops.
        if sleep:
            self.set_sleep(True)
        else:
            # Honour per-agent sleep flags that were persisted from a previous
            # run, but don't flip the global default on.
            with self._lock:
                resume_names = [
                    n for n, a in self._agents.items() if a.sleep_enabled
                ]
            for n in resume_names:
                self._ensure_sleep_thread(n)

        _emit("factory_started", {"workers": workers, "sleep": bool(sleep)})
        self._log.info(
            "AI Factory started with %d workers (sleep=%s)", workers, bool(sleep),
        )

    def stop(self) -> None:
        self._active = False
        # Sleep threads exit on their own next iteration (within ~SLEEP_LOOP_INTERVAL_S).
        _emit("factory_stopped", {})
        self._log.info("AI Factory stopped")

    # ── Status snapshot (for dashboard) ──────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            agents = {
                name: {
                    "model":         a.model,
                    "current_task":  a.current_task,
                    "current_task_id": a.current_task_id,
                    "progress":      a.progress,
                    "paused":        a.paused,
                    "skills":        list(a.skills),
                    "sleep_enabled": a.sleep_enabled,
                    "total_tasks":   a.total_tasks,
                    "failed_tasks":  a.failed_tasks,
                    "vram_used_mb":  self.vram.used_mb(a.gpu_id),
                    "vram_free_mb":  self.vram.free_mb(a.gpu_id),
                    "pending_swap":  list(a.pending_swap) if a.pending_swap else None,
                }
                for name, a in self._agents.items()
            }
        return {
            "active":       self._active,
            "queue_depth":  self._task_q.qsize(),
            "agents":       agents,
            "gpu_ratio":    round(self.vram.usage_ratio() * 100, 1),
            "sleep_default": self._sleep_enabled_default,
            "recent":       self.db.recent_completions(10),
        }

    # ── Task listing (for the headless CLI) ──────────────────────────────────

    def list_tasks(
        self,
        state: str = "all",
        limit: int = 50,
        agent: Optional[str] = None,
    ) -> dict:
        """Return tasks across states.

        state ∈ {"pending", "running", "completed", "all"}
        """
        state = state.lower()
        out: dict = {"state": state, "limit": limit, "pending": [],
                     "running": [], "completed": []}

        if state in ("pending", "all"):
            # Snapshot the priority queue without disturbing ordering.
            with self._lock:
                pending = list(self._task_q.queue)
            pending_dicts = [
                {
                    "task_id":          t.task_id,
                    "priority":         t.priority,
                    "created_at":       t.created_at,
                    "prompt":           t.prompt[:200],
                    "agent_name":       t.agent_name,
                    "autonomous":       t.autonomous,
                    "delegation_depth": t.delegation_depth,
                }
                for t in pending
                if (agent is None or t.agent_name == agent)
            ]
            pending_dicts.sort(key=lambda r: (r["priority"], r["created_at"]))
            out["pending"] = pending_dicts[:limit]

        if state in ("running", "all"):
            with self._lock:
                out["running"] = [
                    {
                        "task_id":   a.current_task_id,
                        "agent":     a.name,
                        "prompt":    a.current_task,
                        "started":   a.task_start,
                        "progress":  a.progress,
                    }
                    for a in self._agents.values()
                    if a.current_task != "idle"
                    and (agent is None or a.name == agent)
                ]

        if state in ("completed", "all"):
            recent = self.db.recent_completions(limit)
            if agent is not None:
                recent = [r for r in recent if r.get("agent_name") == agent]
            out["completed"] = recent[:limit]

        return out


# ── Module-level singleton ────────────────────────────────────────────────────

_factory: Optional[AIFactory] = None


def get_factory(**kwargs: Any) -> AIFactory:
    """Return the module-level AIFactory singleton.

    On first call, kwargs are forwarded to AIFactory.__init__ for configuration
    (e.g. min_priority, max_retries, gpu_warn_threshold).  Subsequent calls
    ignore kwargs and return the already-created instance.
    """
    global _factory
    if _factory is None:
        _factory = AIFactory(**kwargs)
    return _factory
