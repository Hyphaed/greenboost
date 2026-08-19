"""Concurrency-safety must be fail-closed, and batching must preserve order."""
import pytest

from greenboost_cli.instruments.concurrency import (
    is_concurrency_safe, partition_tool_calls, _bash_is_read_only,
)


@pytest.mark.parametrize("name", ["Read", "Grep", "Glob", "Semble",
                                  "WebFetch", "TodoRead", "MemoryRead"])
def test_read_only_instruments_are_safe(name):
    assert is_concurrency_safe(name, {}) is True


@pytest.mark.parametrize("name", ["Write", "Edit", "TodoWrite", "MemoryWrite",
                                  "Screenshot", "mcp__forge3d__optimize_mesh",
                                  "TotallyUnknownTool", ""])
def test_everything_else_is_unsafe(name):
    assert is_concurrency_safe(name, {}) is False


@pytest.mark.parametrize("cmd", [
    "ls -la /tmp",
    "cat a.txt | head -20",
    "grep -rn foo src/ | sort | uniq -c",
    "find . -name '*.py'",
    "nvidia-smi --query-gpu=temperature.gpu --format=csv",
])
def test_read_only_bash_is_safe(cmd):
    assert is_concurrency_safe("Bash", {"command": cmd}) is True


@pytest.mark.parametrize("cmd", [
    "rm -rf /tmp/x",                  # destructive binary
    "echo hi > out.txt",              # redirect
    "cat a.txt && rm b.txt",          # chaining
    "sudo lsmod",                     # privilege
    "python3 script.py",              # unknown binary
    "sed -i 's/a/b/' f.txt",          # in-place edit
    "find . -name '*.tmp' -delete",   # find that writes
    "find . -exec rm {} ;",           # find that executes
    "ls `whoami`",                    # command substitution
    "ls $(whoami)",                   # command substitution
    "cat 'unbalanced",                # unparseable
    "",                               # empty
])
def test_writing_or_unprovable_bash_is_unsafe(cmd):
    assert is_concurrency_safe("Bash", {"command": cmd}) is False


def test_bash_without_a_string_command_is_unsafe():
    assert is_concurrency_safe("Bash", {}) is False
    assert is_concurrency_safe("Bash", {"command": None}) is False
    assert is_concurrency_safe("Bash", None) is False


def _tc(name, **inp):
    return {"name": name, "input": inp}


def test_consecutive_safe_calls_group_together():
    calls = [_tc("Read"), _tc("Grep"), _tc("Glob")]
    batches = partition_tool_calls(calls)
    assert len(batches) == 1
    assert batches[0][0] is True and len(batches[0][1]) == 3


def test_a_write_splits_the_reads_around_it():
    """The ordering guarantee: reads must never jump a write."""
    calls = [_tc("Read"), _tc("Write"), _tc("Grep")]
    batches = partition_tool_calls(calls)
    assert [(s, len(b)) for s, b in batches] == [(True, 1), (False, 1), (True, 1)]
    # and the original order is fully recoverable
    flat = [tc for _, b in batches for tc in b]
    assert flat == calls


def test_unsafe_calls_never_group():
    calls = [_tc("Write"), _tc("Edit"), _tc("Bash", command="rm -rf x")]
    batches = partition_tool_calls(calls)
    assert len(batches) == 3
    assert all(s is False for s, _ in batches)


def test_order_is_preserved_across_a_mixed_batch():
    calls = [_tc("Read"), _tc("Grep"), _tc("Write"), _tc("Read"),
             _tc("Bash", command="ls"), _tc("Bash", command="rm x")]
    batches = partition_tool_calls(calls)
    assert [tc for _, b in batches for tc in b] == calls
    assert [(s, len(b)) for s, b in batches] == [
        (True, 2), (False, 1), (True, 2), (False, 1)]


def test_empty_input():
    assert partition_tool_calls([]) == []


# ── MCP tools: the server's declaration, never the tool's name ──────────────

class _FakeRegistry:
    def __init__(self, schemas):
        self.tool_schemas = schemas


def test_mcp_tool_is_serial_when_the_server_declares_nothing():
    reg = _FakeRegistry([{"name": "mcp__gb__dataflux_summary"}])
    assert is_concurrency_safe("mcp__gb__dataflux_summary", {}, reg) is False


def test_mcp_tool_overlaps_when_the_server_declares_it_read_only():
    reg = _FakeRegistry([{"name": "mcp__gb__dataflux_summary",
                          "annotations": {"readOnlyHint": True}}])
    assert is_concurrency_safe("mcp__gb__dataflux_summary", {}, reg) is True


def test_a_destructive_read_is_still_serial():
    """readOnlyHint plus destructiveHint is a contradictory declaration —
    resolve it the safe way."""
    reg = _FakeRegistry([{"name": "mcp__gb__x",
                          "annotations": {"readOnlyHint": True,
                                          "destructiveHint": True}}])
    assert is_concurrency_safe("mcp__gb__x", {}, reg) is False


def test_no_registry_means_serial():
    assert is_concurrency_safe("mcp__gb__dataflux_summary", {}) is False


def test_write_tool_on_a_read_only_server_is_still_serial():
    reg = _FakeRegistry([
        {"name": "mcp__gb__dataflux_summary", "annotations": {"readOnlyHint": True}},
        {"name": "mcp__gb__factory_submit"},
    ])
    assert is_concurrency_safe("mcp__gb__factory_submit", {}, reg) is False
