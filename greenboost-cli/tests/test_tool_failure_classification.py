"""Tool failures are classified, and the loop guard treats them differently.

Adapted from NemoClaw's validation-recovery classifier, which separates the
"transport" failures it retries from validation failures it reports. GreenBoost
counted every failure identically: a Bash timeout advanced the consecutive-error
guard exactly like "file not found", so a run of transient hiccups could end a
long agentic turn while a genuinely stuck sequence got the same budget.

On this hardware a wasted turn is minutes, not milliseconds, so retrying a
semantic failure verbatim is expensive and pointless , which is why every
semantic message now names the tool to call instead.
"""
import pytest

from greenboost_cli.instruments.handlers import classify_tool_failure


@pytest.mark.parametrize("result,kind,retry", [
    ("Wrote 12 lines to x.py",                                  "ok",        False),
    ("",                                                        "ok",        False),
    ("Error: timed out after 120s",                             "transient", True),
    ("Error: device or resource busy",                          "transient", True),
    ("Error: connection reset by peer",                         "transient", True),
    ("Error: file not found: a.py",                             "semantic",  False),
    ("Error: /x is a directory, and Read only reads files.",    "semantic",  False),
    ("Error: old_string not found in x.py.",                    "semantic",  False),
    ("Blocked by plan mode: not a read-only operation",         "denied",    False),
])
def test_classification(result, kind, retry):
    assert classify_tool_failure(result) == (kind, retry)


def test_an_unrecognised_error_is_never_retried_blindly():
    """Unknown failures default to semantic: repeating a call we do not
    understand risks re-running a side effect."""
    kind, retry = classify_tool_failure("Error: something nobody anticipated")
    assert kind == "semantic" and retry is False


def test_success_text_containing_the_word_error_is_not_a_failure():
    """A grep hit for "error" in output must not be read as a failed call."""
    kind, _ = classify_tool_failure("12 results\nsrc/a.py: raise ValueError('error')")
    assert kind == "ok"


# ── The actionable messages the classification depends on ────────────────────

def test_read_on_a_directory_names_the_recovery(tmp_path):
    from greenboost_cli.instruments.handlers import handle_read

    out = handle_read(str(tmp_path))
    assert "is a directory" in out
    assert "Bash(ls" in out and "Glob(" in out, "no recovery named"


def test_missing_file_suggests_a_near_match(tmp_path):
    from greenboost_cli.instruments.handlers import handle_read

    (tmp_path / "README.md").write_text("hi")
    out = handle_read(str(tmp_path / "READMEE.md"))
    assert "Did you mean" in out and "README.md" in out


def test_edit_separates_whitespace_mismatch_from_absent(tmp_path):
    """The two causes need different fixes, so they must read differently."""
    from greenboost_cli.instruments.handlers import handle_edit

    f = tmp_path / "x.py"
    f.write_text("def go():\n        return 1\n")

    ws = handle_edit(str(f), "def go():\n    return 1", "x")
    assert "not BYTE-for-byte" in ws and "verbatim" in ws

    absent = handle_edit(str(f), "nothing like this exists", "x")
    assert "not found" in absent and "BYTE-for-byte" not in absent


def test_turn_correlation_is_omitted_when_unset():
    """A turn id is never invented , the field is simply absent outside a turn."""
    from greenboost_cli.instruments import dispatcher as D

    D.set_turn_id("")
    assert D._CURRENT_TURN_ID == ""
    D.set_turn_id("t-1")
    assert D._CURRENT_TURN_ID == "t-1"
    D.set_turn_id("")
