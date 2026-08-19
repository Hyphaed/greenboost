"""
Unified advisory framework for GreenBoost.

Ported from NemoClaw's advisory system (src/lib/advisories/), generalized for
GreenBoost's lifecycle phases and producer functions.

Advisory shapes are stable (id, phase, severity) and deduplicable across
multiple producer runs. The registry machinery ensures no duplicate IDs and
preserves resume-safe cached results.
"""

from dataclasses import dataclass, asdict as dataclass_asdict
from enum import Enum
from typing import Callable, Optional
import json

__all__ = [
    "AdvisorySeverity",
    "AdvisoryPhase",
    "Advisory",
    "AdvisoryCheck",
    "AdvisoryRunResult",
    "define_registry",
    "run_advisories",
    "blocking_advisories",
    "assert_no_blocking",
    "BlockingAdvisoryError",
    "format_advisories",
]


class AdvisorySeverity(str, Enum):
    """Severity levels shared by advisory producers and presenters."""
    FATAL = "fatal"
    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class AdvisoryPhase(str, Enum):
    """Lifecycle phases in which advisory checks can run."""
    # Initial system checks
    PREFLIGHT_HOST = "preflight.host"
    PREFLIGHT_CLUSTER = "preflight.cluster"

    # Serving phases
    SERVE_PRE = "serve.pre"
    SERVE_POST = "serve.post"

    # Runtime phases
    RUNTIME_TIER = "runtime.tier"
    RUNTIME_QUANT = "runtime.quant"
    RUNTIME_OPTIMIZE = "runtime.optimize"
    RUNTIME_HEALTH = "runtime.health"

    # Cluster phases
    CLUSTER_DISPATCH = "cluster.dispatch"

    # Installation
    INSTALL = "install"


class AdvisoryKind(str, Enum):
    """Remediation interaction required from an operator."""
    MANUAL = "manual"
    SUDO = "sudo"
    INFO = "info"


@dataclass(frozen=True)
class Advisory:
    """A structured, stable diagnostic produced by an advisory check."""
    id: str
    severity: AdvisorySeverity
    phase: AdvisoryPhase
    title: str
    reason: str
    commands: tuple[str, ...] = ()
    docs_url: Optional[str] = None
    resume_safe: bool = False
    kind: Optional[AdvisoryKind] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict, preserving Enum string values."""
        d = dataclass_asdict(self)
        d['severity'] = self.severity.value
        d['phase'] = self.phase.value
        if self.kind:
            d['kind'] = self.kind.value
        return d


@dataclass(frozen=True)
class AdvisoryCheck:
    """A pure advisory check and the metadata that controls its execution."""
    id: str
    phase: AdvisoryPhase
    severity: AdvisorySeverity
    resume_safe: bool
    check: Callable[..., Optional[Advisory]]
    skip_if: Optional[Callable[..., bool]] = None


@dataclass(frozen=True)
class AdvisoryRunResult:
    """Structured output used to persist safe results across resume boundaries."""
    advisories: tuple[Advisory, ...]
    results: dict[str, Optional[Advisory]]  # id -> Advisory or None
    executed_check_ids: tuple[str, ...]
    reused_check_ids: tuple[str, ...]


def define_registry(checks: list[AdvisoryCheck]) -> tuple[AdvisoryCheck, ...]:
    """
    Creates an explicit, immutable advisory registry with globally unique IDs.

    Raises:
        ValueError: if any duplicate check.id is found
    """
    ids = set()
    for check in checks:
        if check.id in ids:
            raise ValueError(f"Duplicate advisory check id '{check.id}'.")
        ids.add(check.id)
    return tuple(checks)


def _can_suppress(advisory: Advisory) -> bool:
    """Returns True if the advisory can be suppressed by the caller."""
    return advisory.severity not in (AdvisorySeverity.FATAL, AdvisorySeverity.BLOCKING)


def run_advisories(
    checks: tuple[AdvisoryCheck, ...],
    context: object = None,
    phase: Optional[AdvisoryPhase | list[AdvisoryPhase]] = None,
    suppressed: Optional[set[str]] = None,
    cached_results: Optional[dict[str, Optional[Advisory]]] = None,
    resuming: bool = False,
) -> AdvisoryRunResult:
    """
    Runs checks in registry order. During resume, cached results are reused only
    for checks that explicitly declare their prior verdict safe to reuse.

    Args:
        checks: Registry of AdvisoryCheck instances (ideally from define_registry)
        context: Context object passed to each check's check() function
        phase: Optional phase or list of phases to filter by; None = run all
        suppressed: Set of advisory IDs that should be filtered from output
        cached_results: Map of id -> Advisory|None from a prior run
        resuming: If True, reuse cached_results for resume_safe checks

    Returns:
        AdvisoryRunResult with advisories list, results map, execution tracking

    Raises:
        ValueError: if duplicate check.id found during iteration
        AssertionError: if any returned advisory mismatches its check's metadata
    """
    # Normalize phase filter
    phase_set = None
    if phase is not None:
        if isinstance(phase, AdvisoryPhase):
            phase_set = {phase}
        else:
            phase_set = set(phase)

    suppressed_ids = suppressed or set()
    results = {}
    advisories = []
    executed_check_ids = []
    reused_check_ids = []
    seen_ids = set()

    for check in checks:
        # Duplicate-ID check
        if check.id in seen_ids:
            raise ValueError(f"Duplicate advisory check id '{check.id}'.")
        seen_ids.add(check.id)

        # Phase filter
        if phase_set is not None and check.phase not in phase_set:
            continue

        # Skip-if predicate
        if check.skip_if and check.skip_if(context):
            results[check.id] = None
            continue

        # Resume cache reuse decision
        can_reuse = (
            resuming
            and check.resume_safe
            and cached_results is not None
            and check.id in cached_results
        )

        if can_reuse:
            advisory = cached_results[check.id]
            reused_check_ids.append(check.id)
        else:
            advisory = check.check(context)
            executed_check_ids.append(check.id)

        # Assert metadata matches
        if advisory is not None:
            _assert_advisory_matches_check(check, advisory)

        results[check.id] = advisory

        # Add to output if not suppressed
        if advisory is not None:
            if check.id not in suppressed_ids or not _can_suppress(advisory):
                advisories.append(advisory)

    return AdvisoryRunResult(
        advisories=tuple(advisories),
        results=results,
        executed_check_ids=tuple(executed_check_ids),
        reused_check_ids=tuple(reused_check_ids),
    )


def _assert_advisory_matches_check(check: AdvisoryCheck, advisory: Advisory) -> None:
    """
    Verify that an advisory returned by a check matches its static metadata.

    This guards against copy-paste bugs where a check's declared id/phase/
    severity/resumeSafe diverges from what its check() function actually returns.
    """
    mismatches = []
    if advisory.id != check.id:
        mismatches.append(f"id '{advisory.id}' != '{check.id}'")
    if advisory.phase != check.phase:
        mismatches.append(f"phase '{advisory.phase.value}' != '{check.phase.value}'")
    if advisory.severity != check.severity:
        mismatches.append(f"severity '{advisory.severity.value}' != '{check.severity.value}'")
    if advisory.resume_safe != check.resume_safe:
        mismatches.append(f"resume_safe {advisory.resume_safe} != {check.resume_safe}")

    if mismatches:
        raise AssertionError(
            f"Advisory check '{check.id}' returned mismatched metadata: {', '.join(mismatches)}."
        )


def blocking_advisories(advisories: list[Advisory] | tuple[Advisory, ...]) -> list[Advisory]:
    """Returns the findings that prohibit the caller from continuing."""
    return [
        a for a in advisories
        if a.severity in (AdvisorySeverity.FATAL, AdvisorySeverity.BLOCKING)
    ]


class BlockingAdvisoryError(Exception):
    """Error raised when a caller attempts to continue past blocking advisories."""
    def __init__(self, advisories: list[Advisory] | tuple[Advisory, ...]):
        self.advisories = list(advisories)
        count = len(self.advisories)
        msg = f"Blocked by {count} advisory finding{'s' if count != 1 else ''}."
        super().__init__(msg)


def assert_no_blocking(advisories: list[Advisory] | tuple[Advisory, ...]) -> None:
    """
    Throws BlockingAdvisoryError when any fatal or blocking finding is present.
    """
    blocking = blocking_advisories(advisories)
    if blocking:
        raise BlockingAdvisoryError(blocking)


def format_advisories(
    advisories: list[Advisory] | tuple[Advisory, ...],
    fmt: str = "console",
) -> str:
    """
    Formats advisories for display or logging.

    Args:
        advisories: List or tuple of Advisory objects
        fmt: Format type; "console" or "json"

    Returns:
        Formatted string
    """
    if fmt == "json":
        return json.dumps([a.to_dict() for a in advisories], indent=2)

    # Console format
    lines = []
    for advisory in advisories:
        lines.append(
            f"[{advisory.severity.value.upper()}] {advisory.title} ({advisory.id})"
        )
        lines.append(f"  {advisory.reason}")
        for cmd in advisory.commands:
            lines.append(f"  Run: {cmd}")
        if advisory.docs_url:
            lines.append(f"  More: {advisory.docs_url}")
        lines.append("")

    return "\n".join(lines)
