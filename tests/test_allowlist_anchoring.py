"""Content-anchored allowlist entries , `path:~<sha8>`.

Line-numbered allowlist entries drift every time anything is inserted above
them, and drift here is not cosmetic: the entry keeps matching a line NUMBER
that now holds unrelated code, so it silently sanctions something nobody
reviewed while the line it was written for goes unguarded.

That happened for real. On 2026-08-21 a four-entry group in
secrets_reviewed.txt was found pointing at ordinary prose , it had been
papered over by ADDING a second group instead of repairing the first, so two
four-entry groups existed for a file containing exactly four matching lines.
The hardware allowlist drifted four separate times in one session's edits.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks"))
from lib import content_anchor, is_allowlisted            # noqa: E402

LINE = "    Measured on this box 2026-08-18: 15.86 GB of weights against 12227 MiB"


def test_an_anchored_entry_matches_wherever_the_line_moved_to():
    pat = [f"gb_semantics.py:~{content_anchor(LINE)}"]
    # Same content, wildly different line numbers , both allowed.
    assert is_allowlisted("gb_semantics.py", 12, pat, LINE)
    assert is_allowlisted("gb_semantics.py", 9999, pat, LINE)


def test_reindenting_does_not_invalidate_an_anchor():
    """Whitespace-normalised on purpose: re-indenting a docstring does not
    change what it says, and an allowlist that fails on reflow trains people
    to stop maintaining it."""
    pat = [f"gb_semantics.py:~{content_anchor(LINE)}"]
    assert is_allowlisted("gb_semantics.py", 1, pat, "        " + LINE.strip())


def test_changing_what_the_line_says_revokes_the_sanction():
    """The whole point. A reviewed literal stays reviewed; a DIFFERENT literal
    on the same line is a new finding a human has not seen."""
    pat = [f"gb_semantics.py:~{content_anchor(LINE)}"]
    assert not is_allowlisted("gb_semantics.py", 1, pat, LINE.replace("15.86", "99.99"))


def test_an_anchor_is_scoped_to_its_file():
    pat = [f"gb_semantics.py:~{content_anchor(LINE)}"]
    assert not is_allowlisted("gb_synapse.py", 1, pat, LINE)


def test_line_numbered_entries_still_work():
    """Backward compatibility , most entries are still by line."""
    assert is_allowlisted("x.py", 5, ["x.py:5"])
    assert not is_allowlisted("x.py", 6, ["x.py:5"])


def test_whole_file_and_glob_entries_still_work():
    assert is_allowlisted("third_party/llama.cpp/a.c", 1, ["third_party/*"])
    assert is_allowlisted("x.py", 1, ["x.py"])


def test_no_line_text_means_anchors_simply_do_not_match():
    """A caller that has not been updated to pass the line gets the old
    behaviour, never a spurious allow."""
    pat = [f"gb_semantics.py:~{content_anchor(LINE)}"]
    assert not is_allowlisted("gb_semantics.py", 1, pat)


def test_every_anchored_entry_in_the_repo_still_resolves():
    """A stale anchor sanctions nothing, which is safe , but it is also dead
    weight that hides the fact the reviewed line is gone. Catch it here."""
    repo = Path(__file__).resolve().parent.parent
    for name in ("hardware.txt", "secrets_reviewed.txt"):
        allow = repo / "checks" / "allowlists" / name
        if not allow.is_file():
            continue
        for raw in allow.read_text().splitlines():
            entry = raw.split("#", 1)[0].strip()
            if ":~" not in entry:
                continue
            rel, anchor = entry.split(":~", 1)
            target = repo / rel
            assert target.is_file(), f"{name}: {rel} does not exist"
            hits = sum(1 for ln in target.read_text(errors="replace").splitlines()
                       if content_anchor(ln) == anchor.strip())
            assert hits >= 1, (
                f"{name}: anchor {anchor} no longer matches any line in {rel} , "
                f"the reviewed content changed or was deleted; re-review and update")
