"""Tests for gb_wait.py — testable polling primitive with exponential backoff."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gb_wait import wait_until, WaitTimeoutError


class FakeClock:
    """Injectable fake clock for testing wait_until without real time."""
    def __init__(self):
        self.current_time = 0.0
        self.sleep_calls = []

    def now(self):
        return self.current_time

    def sleep(self, duration):
        self.sleep_calls.append(duration)
        self.current_time += duration


def test_condition_true_immediately():
    """Condition is true on first check — returns immediately, no sleep."""
    clock = FakeClock()
    condition_calls = [0]

    def condition():
        condition_calls[0] += 1
        return True

    wait_until(condition, deadline_s=10.0, now=clock.now, sleep=clock.sleep)
    assert condition_calls[0] == 1
    assert clock.sleep_calls == []


def test_condition_true_on_attempt_n():
    """Condition becomes true on attempt N — correct number of sleeps."""
    clock = FakeClock()
    condition_calls = [0]

    def condition():
        condition_calls[0] += 1
        return condition_calls[0] >= 5  # True on 5th attempt

    wait_until(condition, deadline_s=100.0, initial_interval_s=1.0,
               max_interval_s=10.0, backoff_factor=1.5, now=clock.now,
               sleep=clock.sleep)

    assert condition_calls[0] == 5
    # 4 sleeps: 1.0, 1.5, 2.25, 3.375
    assert len(clock.sleep_calls) == 4
    assert clock.sleep_calls[0] == 1.0
    assert clock.sleep_calls[1] == 1.5
    assert clock.sleep_calls[2] == 2.25
    assert clock.sleep_calls[3] == 3.375
    # Total time: 1.0 + 1.5 + 2.25 + 3.375 = 8.125
    assert clock.current_time == pytest.approx(8.125)


def test_deadline_timeout_raises():
    """Condition never true — raises WaitTimeoutError at deadline."""
    clock = FakeClock()
    call_count = [0]

    def condition():
        call_count[0] += 1
        return False

    with pytest.raises(WaitTimeoutError) as exc_info:
        wait_until(condition, deadline_s=5.0, initial_interval_s=1.0,
                   now=clock.now, sleep=clock.sleep)

    assert "deadline of 5.0s elapsed" in str(exc_info.value)
    assert clock.current_time >= 5.0


def test_max_attempts_limits_retries():
    """max_attempts bounds the number of attempts independently of deadline."""
    clock = FakeClock()
    call_count = [0]

    def condition():
        call_count[0] += 1
        return False

    with pytest.raises(WaitTimeoutError) as exc_info:
        wait_until(condition, deadline_s=100.0, initial_interval_s=1.0,
                   max_attempts=3, now=clock.now, sleep=clock.sleep)

    assert "max_attempts (3) exhausted" in str(exc_info.value)
    assert call_count[0] == 3
    # 2 sleeps (between 3 attempts)
    assert len(clock.sleep_calls) == 2


def test_fixed_interval_no_backoff():
    """backoff_factor=1.0 means no exponential backoff — fixed intervals."""
    clock = FakeClock()
    call_count = [0]

    def condition():
        call_count[0] += 1
        return call_count[0] >= 6

    wait_until(condition, deadline_s=100.0, initial_interval_s=2.0,
               max_interval_s=2.0, backoff_factor=1.0, now=clock.now,
               sleep=clock.sleep)

    # 5 sleeps of 2.0 each
    assert clock.sleep_calls == [2.0] * 5
    assert clock.current_time == 10.0


def test_max_interval_caps_backoff():
    """Exponential backoff is capped at max_interval."""
    clock = FakeClock()
    call_count = [0]

    def condition():
        call_count[0] += 1
        return False

    # This will timeout, but we can inspect the sleep calls
    try:
        wait_until(condition, deadline_s=100.0, initial_interval_s=1.0,
                   max_interval_s=3.0, backoff_factor=2.0, max_attempts=6,
                   now=clock.now, sleep=clock.sleep)
    except WaitTimeoutError:
        pass

    # Intervals should grow but be capped at 3.0: 1.0, 2.0, 3.0, 3.0, 3.0
    assert clock.sleep_calls == [1.0, 2.0, 3.0, 3.0, 3.0]


def test_condition_exception_propagates():
    """Exception raised by condition() is propagated, not caught."""
    clock = FakeClock()
    call_count = [0]

    class CustomError(Exception):
        pass

    def condition():
        call_count[0] += 1
        if call_count[0] == 2:
            raise CustomError("Something went wrong")
        return False

    with pytest.raises(CustomError):
        wait_until(condition, deadline_s=10.0, initial_interval_s=1.0,
                   now=clock.now, sleep=clock.sleep)

    assert call_count[0] == 2  # Called twice: once returns False, twice raises


def test_deadline_binds_before_max_attempts():
    """When both deadline and max_attempts are set, deadline can bind first."""
    clock = FakeClock()
    call_count = [0]

    def condition():
        call_count[0] += 1
        return False

    # Deadline is 3 seconds, max_attempts is 100.
    # With 1-second sleeps, we'll hit the deadline before max_attempts.
    with pytest.raises(WaitTimeoutError) as exc_info:
        wait_until(condition, deadline_s=3.0, initial_interval_s=1.0,
                   max_attempts=100, now=clock.now, sleep=clock.sleep)

    assert "deadline of 3.0s elapsed" in str(exc_info.value)
    # We make attempts until the deadline, then raise.
    # Roughly 4 attempts (check, sleep 1, check, sleep 1, check, sleep 1, check, deadline).
    assert call_count[0] <= 5


def test_max_attempts_binds_before_deadline():
    """When both deadline and max_attempts are set, max_attempts can bind first."""
    clock = FakeClock()
    call_count = [0]

    def condition():
        call_count[0] += 1
        return False

    # Deadline is 1000 seconds, max_attempts is 3.
    # We'll hit max_attempts first.
    with pytest.raises(WaitTimeoutError) as exc_info:
        wait_until(condition, deadline_s=1000.0, initial_interval_s=1.0,
                   max_attempts=3, now=clock.now, sleep=clock.sleep)

    assert "max_attempts (3) exhausted" in str(exc_info.value)
    assert call_count[0] == 3


def test_sleep_duration_clamped_at_deadline():
    """Final sleep is clamped to not overshoot the deadline."""
    clock = FakeClock()
    call_count = [0]

    def condition():
        call_count[0] += 1
        return False

    with pytest.raises(WaitTimeoutError):
        # Deadline 5 seconds, sleep 2 seconds per attempt.
        # Attempts: check (0s), sleep 2 (2s), check (2s), sleep 2 (4s),
        # check (4s), sleep 1 (5s, clamped from 2s), then deadline exceeded.
        wait_until(condition, deadline_s=5.0, initial_interval_s=2.0,
                   max_interval_s=2.0, backoff_factor=1.0, now=clock.now,
                   sleep=clock.sleep)

    # The last sleep should be clamped to avoid overshooting
    assert clock.sleep_calls[-1] <= 1.0  # clamped from 2.0


def test_no_deadline_no_max_attempts_raises_error():
    """Providing neither deadline_s nor max_attempts raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        wait_until(lambda: False, deadline_s=float("inf"), max_attempts=None)

    assert "requires either a finite deadline_s or max_attempts" in str(exc_info.value)


def test_real_time_sleep_respects_deadlines():
    """Smoke test with real time.sleep and time.monotonic (no injection)."""
    import time

    start = time.monotonic()
    call_count = [0]

    def condition():
        call_count[0] += 1
        return call_count[0] >= 3

    # Use real time, 0.01s sleep intervals, 1s deadline
    wait_until(condition, deadline_s=1.0, initial_interval_s=0.01)

    elapsed = time.monotonic() - start
    # Should complete quickly (3 attempts with ~2 sleeps of 0.01s each ≈ 0.02s)
    # but definitely within the 1s deadline.
    assert elapsed < 1.0
    assert call_count[0] == 3


def test_error_message_includes_context():
    """WaitTimeoutError messages include relevant context."""
    clock = FakeClock()

    def condition():
        return False

    try:
        wait_until(condition, deadline_s=5.0, max_attempts=10, now=clock.now,
                   sleep=clock.sleep)
    except WaitTimeoutError as e:
        # Should mention deadline when it expires first
        assert "5.0s" in str(e)
