"""A worker must return one result per item it was given.

The soft failure this closes is the one GreenBoost's own standing rule tells
operators to work around by hand:

    "feeder soft-failures may NOT appear as dataflux errors , the process runs,
     produces nothing, reports success. Cross-check item counts per node against
     expected outputs on disk."

That workaround existed because nothing reconciled the counts in code. Both
collection paths do `for j, r in enumerate(out[offset:offset + len(b)])`, so a
short `out` simply yields fewer iterations: the unfilled result slots keep their
initial value, `_mark_done(len(batch))` reports the batch complete, and the
caller gets a list with holes and no error anywhere. A worker that produced
NOTHING was indistinguishable from one that produced everything.
"""
import pytest

import gb_cluster


@pytest.fixture()
def emitted(monkeypatch):
    seen = []
    monkeypatch.setattr(gb_cluster, "_df_emit",
                        lambda *a, **k: seen.append((a, k)))
    return seen


def test_a_complete_batch_is_silent(emitted):
    assert gb_cluster._reconcile_chunk("r", "n", "l", "chunk_local",
                                       [1, 2, 3], [1, 2, 3]) == 0
    assert emitted == [], "a correct batch must not emit anything"


@pytest.mark.parametrize("out,missing", [([1], 2), ([], 3), (None, 3)])
def test_a_short_return_is_reported(emitted, out, missing):
    got = gb_cluster._reconcile_chunk("r", "n", "l", "chunk_remote", [1, 2, 3], out)
    assert got == missing
    assert len(emitted) == 1, "a shortfall must emit exactly one error event"
    _args, kw = emitted[0]
    assert kw["status"] == "error"
    assert kw["dispatched_items"] == 3
    assert kw["returned_items"] == (len(out) if out is not None else 0)
    assert kw["missing_items"] == missing
    assert "mismatch" in kw["error"]


def test_the_empty_case_is_the_one_that_mattered(emitted):
    """'Ran and produced nothing' is the documented failure. It must be loud."""
    assert gb_cluster._reconcile_chunk("r", "omen", "gen", "chunk_remote",
                                       list(range(8)), []) == 8
    _args, kw = emitted[0]
    assert kw["returned_items"] == 0 and kw["missing_items"] == 8


def test_accounting_never_breaks_a_dispatch(monkeypatch):
    """Observation must not fail the work it observes."""
    def _boom(*a, **k):
        raise RuntimeError("dataflux down")
    monkeypatch.setattr(gb_cluster, "_df_emit", _boom)
    assert gb_cluster._reconcile_chunk("r", "n", "l", "chunk_local", [1, 2], []) == 0


def test_an_over_long_return_is_also_flagged(emitted):
    """More results than items is equally wrong, and equally silent before."""
    assert gb_cluster._reconcile_chunk("r", "n", "l", "chunk_local",
                                       [1, 2], [1, 2, 3]) == -1
    assert len(emitted) == 1


def test_every_collection_path_reconciles():
    """Guard: the check must be wired at all three sites, not one."""
    from pathlib import Path
    src = (Path(gb_cluster.__file__)).read_text()
    calls = src.count("_reconcile_chunk(run_id")
    assert calls >= 4, f"expected the definition + 3 call sites, found {calls}"
