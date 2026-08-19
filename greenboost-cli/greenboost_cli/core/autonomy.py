"""Unattended operation: keep working, answer own questions, record it all.

The problem this solves
-----------------------
A turn ends when the model stops emitting tool calls. That is usually right,
but not always: on 2026-08-19 a 48-minute session ended with the model writing
"I'll make those script fixes now." and then producing no tool call. Nothing
was wrong, nothing was finished, and the prompt came back to a user who had
not asked for it. Overnight that is the difference between eight hours of work
and eight hours of an idle prompt.

So there are two questions to answer after every turn:

1. Did the model INTEND to keep going? Stated intent with no tool call is the
   signature of the stall above.
2. Is there work left on the record? A pending todo is objective evidence,
   and it outranks any reading of the prose.

Everything the run does unattended is written to a journal, because a mode
that acts on the user's behalf while they sleep is only acceptable if they can
audit every decision in the morning , which tools ran, which skills, which
questions were auto-answered and why.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

#: Consecutive auto-continues before the run stops on its own. Generous , this
#: is a runaway brake, not a work budget , but never infinite: an unattended
#: loop with no ceiling is how a night's tokens disappear into one stuck task.
DEFAULT_MAX_AUTO_CONTINUES = 200

#: A continuation that produces fewer new output tokens than this did not do
#: enough to count as work. The threshold is a judgement call, not a measured
#: constant: 500 tokens is roughly "wrote a paragraph or called a tool with
#: real arguments", and comfortably above the ~50 tokens a turn spends saying
#: it is about to do something.
STALL_DELTA_TOKENS = 500

#: How many continuations must have happened before the stall check is allowed
#: to fire. Early turns are legitimately short (reading a file, checking a
#: status) and must never be mistaken for a stall.
STALL_MIN_CONTINUES = 3

#: Journal entries kept. At a few tool calls a minute a multi-day run produces
#: hundreds of thousands; this keeps the report readable and the memory flat.
#: Overflow is COUNTED, not hidden , a report that silently omits the first two
#: days is worse than one that says how much it dropped.
MAX_JOURNAL_ENTRIES = 5000

#: Forward intent: the model says it is about to act. Anchored to the start of
#: a line so a retrospective "I'll leave that to you" mid-paragraph does not
#: count. Kept deliberately small , a pending todo is the stronger signal and
#: is checked first.
_INTENT = re.compile(
    r"^\s*(?:[-*]\s*)?(?:i'?ll|i will|i'?m going to|let me|now i|next[,:]|"
    r"proceeding|continuing|i'?ll now|going to)(?:\b|(?<=[,:]))",
    re.IGNORECASE | re.MULTILINE,
)

#: Explicit hand-back. Beats intent when both appear , the model is asking.
_HANDBACK = re.compile(
    r"(?:let me know|what would you like|shall i|would you like me|"
    r"which (?:one|option) |waiting for (?:your|you)|your call|"
    r"do you want me to|should i (?:proceed|continue))",
    re.IGNORECASE,
)

#: Finished. Beats intent; does not beat a pending todo.
_DONE = re.compile(
    r"(?:all (?:done|set)|everything (?:is )?(?:done|working|passing)|"
    r"task (?:is )?complete|finished\b|nothing (?:else|more) to do|"
    r"no further (?:work|changes) )",
    re.IGNORECASE,
)


@dataclass
class JournalEntry:
    ts: float
    kind: str                     # tool | skill | question | continue | stop
    detail: dict = field(default_factory=dict)


@dataclass
class AutonomyState:
    """Per-session unattended-operation state."""

    #: Keep working when a turn ends with work outstanding. Default ON , the
    #: owner's stated default; ctrl+n turns it off.
    nonstop: bool = True
    #: Answer the model's own AskUserQuestion instead of blocking for a human.
    #: Default OFF: silently choosing on the user's behalf is a bigger step
    #: than continuing work they already asked for, so it is opt-in (ctrl+y).
    auto_answer: bool = False

    max_auto_continues: int = DEFAULT_MAX_AUTO_CONTINUES
    consecutive_continues: int = 0
    #: Output tokens produced since this chain started, and the two deltas the
    #: stall check compares. Not resettable by tool activity , that is the
    #: whole point: a loop that keeps calling one cheap tool looks like
    #: progress to `note_progress` and like a stall to these three numbers.
    chain_output_tokens: int = 0
    last_delta_tokens: int = 0
    last_checked_tokens: int = 0
    #: Continuations in THIS chain. Separate from `consecutive_continues`
    #: precisely because that one is reset by tool activity , the stall check
    #: needs a depth counter a busy-looking loop cannot rewind.
    chain_continues: int = 0
    started_ts: float = field(default_factory=time.time)
    journal: list = field(default_factory=list)
    journal_dropped: int = 0        # entries evicted by the cap, reported honestly

    # ── journal ──────────────────────────────────────────────────────────────
    def record(self, kind: str, **detail) -> None:
        self.journal.append(JournalEntry(time.time(), kind, detail))
        # Unattended-For-Days Must-Rule: one entry per tool call is unbounded
        # over a multi-day run. Drop the OLDEST half when the cap is hit rather
        # than trimming one at a time , halving is O(1) amortised, and the
        # morning-after questions ("what did it just do", "why did it stop")
        # are answered by the recent end, not by hour three.
        if len(self.journal) > MAX_JOURNAL_ENTRIES:
            dropped = len(self.journal) - MAX_JOURNAL_ENTRIES // 2
            self.journal = self.journal[dropped:]
            self.journal_dropped += dropped

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for e in self.journal:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out

    def questions(self) -> list:
        return [e for e in self.journal if e.kind == "question"]

    def note_chain_start(self) -> None:
        """A new user-initiated chain begins: forget the previous one's
        token history so its deltas cannot be read as this chain's stall."""
        self.chain_continues = 0
        self.chain_output_tokens = 0
        self.last_delta_tokens = 0
        self.last_checked_tokens = 0

    def note_output_tokens(self, n: int) -> None:
        """Record a turn's generated tokens. Called once per model turn."""
        if n and n > 0:
            self.chain_output_tokens += int(n)

    def _stalled(self) -> bool:
        """Two consecutive near-empty continuations after a real stretch.

        The runaway ceiling above cannot catch this case on its own, because
        `note_progress()` resets it on every tool result , which is correct
        for a chain that is genuinely working and useless against a chain that
        re-runs one cheap lookup forever. Token delta is the signal that does
        not lie: whatever the model is calling, it has stopped producing.

        Advances the counters on every call, so a chain that stops for another
        reason does not leave a stale delta behind for the next one.
        """
        delta = self.chain_output_tokens - self.last_checked_tokens
        stalled = (
            self.chain_continues >= STALL_MIN_CONTINUES
            and delta < STALL_DELTA_TOKENS
            and self.last_delta_tokens < STALL_DELTA_TOKENS
        )
        self.last_delta_tokens = delta
        self.last_checked_tokens = self.chain_output_tokens
        return stalled

    # ── the decision ─────────────────────────────────────────────────────────
    def decide_continue(self, reply_text: str, pending_todos: int = 0):
        """Should the session keep going? Returns (bool, reason).

        Order matters and encodes the priorities:
        a hard budget stops everything; an explicit question to the user is
        always honoured; a pending todo is objective and beats prose; only then
        is stated intent, and only when the model did not also say it is done.
        """
        if not self.nonstop:
            return False, "non-stop mode is off"
        if self.consecutive_continues >= self.max_auto_continues:
            return False, (f"reached the {self.max_auto_continues}-continue "
                           f"ceiling for one stretch")
        if self._stalled():
            return False, (f"the last continuations produced under "
                           f"{STALL_DELTA_TOKENS} new tokens each , the run "
                           f"is going in circles, not forward")
        text = reply_text or ""
        if _HANDBACK.search(text):
            return False, "the model asked the user a question"
        if pending_todos > 0:
            return True, f"{pending_todos} todo(s) still pending"
        if _DONE.search(text):
            return False, "the model reported the work finished"
        if _INTENT.search(text):
            return True, "the model stated it was about to continue"
        return False, "the turn ended with no stated intent and no open todos"

    def note_continue(self, reason: str) -> None:
        self.consecutive_continues += 1
        self.chain_continues += 1
        self.record("continue", reason=reason, n=self.consecutive_continues)

    def note_progress(self) -> None:
        """A turn did real work, so the runaway counter resets."""
        self.consecutive_continues = 0


#: What gets injected as the next user turn when auto-continuing. Phrased as a
#: nudge rather than a new instruction so it cannot redirect the work.
CONTINUE_PROMPT = (
    "Continue with the work you just described. Do not summarise or ask , "
    "carry on with the next concrete step, and keep going until the task is "
    "actually finished."
)


def choose_answer(question: dict) -> tuple[int, str]:
    """Pick an option for an AskUserQuestion without a human. Returns (idx, why).

    Deliberately simple and predictable rather than clever: an option marked
    "(Recommended)" wins, otherwise the first, which the tool's own contract
    says is where a recommendation belongs. A predictable rule is auditable in
    the morning; a model asked to grade its own options at 3am is not.
    """
    options = question.get("options") or []
    if not options:
        return -1, "the question offered no options"
    for i, opt in enumerate(options):
        if "recommend" in str(opt.get("label", "")).lower():
            return i, "the option is marked Recommended"
    return 0, ("no option was marked Recommended; took the first, which the "
               "tool's contract reserves for the recommended choice")


def render_report(state: AutonomyState, session_title: str = "") -> str:
    """A morning-after summary: what ran, what was decided, what was asked."""
    dur = time.time() - state.started_ts
    h, rem = divmod(int(dur), 3600)
    m = rem // 60
    counts = state.counts()
    tools: dict[str, int] = {}
    skills: dict[str, int] = {}
    for e in state.journal:
        if e.kind == "tool":
            n = e.detail.get("name", "?")
            tools[n] = tools.get(n, 0) + 1
        elif e.kind == "skill":
            n = e.detail.get("name", "?")
            skills[n] = skills.get(n, 0) + 1

    L = [f"# Session report{(' , ' + session_title) if session_title else ''}",
         "",
         f"Ran for {h}h {m:02d}m. "
         f"{counts.get('continue', 0)} automatic continuation(s), "
         f"{counts.get('question', 0)} question(s) answered without a human.",
         ""]

    L.append("## Tools used")
    L += ([f"- {n} x{c}" for n, c in sorted(tools.items(), key=lambda x: -x[1])]
          or ["- none"])
    L.append("")
    L.append("## Skills invoked")
    L += ([f"- {n} x{c}" for n, c in sorted(skills.items(), key=lambda x: -x[1])]
          or ["- none"])
    L.append("")

    L.append("## Questions answered automatically")
    qs = state.questions()
    if not qs:
        L.append("- none")
    for e in qs:
        d = e.detail
        L.append(f"- **{d.get('question', '?')}**")
        L.append(f"  - chose: {d.get('chosen', '?')}")
        L.append(f"  - why: {d.get('why', '?')}")
        if d.get("options"):
            L.append(f"  - other options: "
                     f"{', '.join(str(o) for o in d['options'])}")
    L.append("")

    L.append("## Why the session stopped")
    stops = [e for e in state.journal if e.kind == "stop"]
    L.append(f"- {stops[-1].detail.get('reason', 'unknown')}" if stops
             else "- still running")
    return "\n".join(L)


def export_report(state: AutonomyState, path: "str | Path",
                  session_title: str = "") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_report(state, session_title), encoding="utf-8")
    return p


def export_journal_json(state: AutonomyState, path: "str | Path") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(e) for e in state.journal], indent=2),
                 encoding="utf-8")
    return p


# ── per-process singleton ────────────────────────────────────────────────────
# The REPL, its key bindings, its toolbar and the turn loop all need the same
# state, and they do not share an object graph. One module-level instance is
# the honest way to say "there is exactly one session".

_STATE: "AutonomyState | None" = None


def get_state() -> AutonomyState:
    global _STATE
    if _STATE is None:
        _STATE = AutonomyState()
    return _STATE


def reset_state() -> AutonomyState:
    global _STATE
    _STATE = AutonomyState()
    return _STATE
