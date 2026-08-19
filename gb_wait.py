"""gb_wait.py — Testable polling primitive with exponential backoff.

Ported from NemoClaw's core/wait.ts (Apache-2.0, design reference).
Key design point: `now` and `sleep` are injectable dependencies (default to
real time.monotonic / real time.sleep) so tests can simulate arbitrary waits
without real wall-clock time blocking the test suite.
"""
from __future__ import annotations

import time
from typing import Callable


class WaitTimeoutError(Exception):
    """Raised when wait_until's condition never became true within its deadline
    and/or max_attempts."""
    pass


def wait_until(
    condition: Callable[[], bool],
    deadline_s: float = 10.0,
    initial_interval_s: float = 0.25,
    max_interval_s: float = 5.0,
    backoff_factor: float = 1.5,
    max_attempts: int | None = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll `condition()` with exponential backoff until it returns True,
    the deadline elapses, or max_attempts is exhausted (whichever binds first).

    Raises WaitTimeoutError on timeout/attempt exhaustion. Propagates any
    exception raised by condition() — use a wrapper if you want to treat
    condition() exceptions as "not ready yet".

    Args:
        condition: Callable that returns True when the condition is met.
        deadline_s: Deadline in seconds relative to `now()`. No deadline when
            combined with max_attempts=None (will loop forever).
        initial_interval_s: Initial sleep interval between failed attempts.
        max_interval_s: Maximum sleep interval after backoff is applied.
        backoff_factor: Multiplier applied to interval after each failed attempt.
            Set to 1.0 to use fixed intervals (no exponential backoff).
        max_attempts: Optional cap on number of attempts. None = no attempt limit.
        now: Injected clock (default time.monotonic). Used for deadline
            comparisons — allows tests to pass a fake clock.
        sleep: Injected sleep function (default time.sleep). Allows tests to
            pass a fake that doesn't block.

    Raises:
        WaitTimeoutError: When deadline elapsed or max_attempts exhausted
            without condition returning True.
        Any exception raised by condition() is propagated as-is.

    Example (real code):
        >>> def ready():
        ...     return check_port_open("127.0.0.1", 8080)
        >>> wait_until(ready, deadline_s=5.0)  # Blocks until port is open or 5s passes

    Example (test with fake time):
        >>> fake_time = [0.0]
        >>> def fake_now():
        ...     return fake_time[0]
        >>> def fake_sleep(s):
        ...     fake_time[0] += s
        >>> attempt_count = [0]
        >>> def condition():
        ...     attempt_count[0] += 1
        ...     return attempt_count[0] >= 5  # True on 5th attempt
        >>> # Simulate a long wait in zero wall-clock time:
        >>> wait_until(condition, deadline_s=100.0, initial_interval_s=1.0,
        ...            max_interval_s=10.0, now=fake_now, sleep=fake_sleep)
        >>> assert attempt_count[0] == 5
        >>> assert fake_time[0] == 10.0  # 4 sleeps: 1 + 2 + 3 + 4
    """
    if max_attempts is None and not (0 < deadline_s < float("inf")):
        raise ValueError("wait_until requires either a finite deadline_s or max_attempts")

    start_time = now()
    interval_s = initial_interval_s
    attempts = 0

    while True:
        current_time = now()

        # Check deadline
        if current_time - start_time >= deadline_s:
            raise WaitTimeoutError(
                f"deadline of {deadline_s}s elapsed without condition becoming true "
                f"(after {attempts} attempts)"
            )

        # Check attempt cap
        if max_attempts is not None and attempts >= max_attempts:
            raise WaitTimeoutError(
                f"max_attempts ({max_attempts}) exhausted without condition becoming true"
            )

        attempts += 1

        # Check the condition
        if condition():
            return

        # If we've exhausted attempts after checking condition, raise
        if max_attempts is not None and attempts >= max_attempts:
            raise WaitTimeoutError(
                f"max_attempts ({max_attempts}) exhausted without condition becoming true"
            )

        # Calculate sleep duration
        current_time = now()
        time_remaining = deadline_s - (current_time - start_time)

        # Clamp sleep to not overshoot deadline
        sleep_duration = min(interval_s, time_remaining)
        if sleep_duration > 0:
            sleep(sleep_duration)

        # Apply backoff for next iteration
        interval_s = min(max_interval_s, interval_s * backoff_factor)
