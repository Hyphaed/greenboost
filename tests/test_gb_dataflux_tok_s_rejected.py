"""A tok/s sample the writer rejected must not reach the throughput rollup.

`gb_synapse.record_measured_tok_s()` applies a node/model-derived sanity ceiling.
When a sample exceeds it the event is still emitted — for auditability — but
marked `status="error"` with `error="sample exceeds ... ceiling, dropped"`.
`summarize()` used to aggregate every `tok_s_measured` event regardless, so the
impossible figure became the reported `latest` and dragged `avg` with it.

Observed live 2026-08-17: `dataflux_tok_s` reported latest=21065.2 tok/s for the
serving model against that node's own computed ceiling of 2566.0.
"""
from __future__ import annotations

import gb_dataflux


def _ev(tok_s: float, ts: float, status: str = "ok", **extra) -> dict:
    ev = {
        "kind": "tok_s_measured",
        "node": "host",
        "model": "test-model",
        "source": "proxy",
        "quant": "Q4",
        "ctx": 32768,
        "kv_type": "f16",
        "tok_s": tok_s,
        "status": status,
        "ts": ts,
    }
    ev.update(extra)
    return ev


def _rollup(events: list[dict]) -> dict:
    return gb_dataflux.summarize(events)["tok_s"]


def test_rejected_sample_is_excluded_from_the_rollup() -> None:
    rows = _rollup([
        _ev(5.0, 100.0),
        _ev(21065.2, 200.0, status="error", ceiling_tok_s=2566.0,
            error="sample exceeds node/model-derived sanity ceiling, dropped"),
    ])
    (stats,) = rows.values()
    assert stats["samples"] == 1, "the rejected sample must not be counted"
    assert stats["latest"] == 5.0, "a dropped sample must never become `latest`"
    assert stats["avg"] == 5.0


def test_a_run_of_only_rejected_samples_reports_nothing() -> None:
    rows = _rollup([
        _ev(21065.2, 100.0, status="error"),
        _ev(18355.3, 200.0, status="error"),
    ])
    assert rows == {}, "no valid measurement means no throughput row, not a fake one"


def test_valid_samples_still_aggregate_normally() -> None:
    rows = _rollup([_ev(4.0, 100.0), _ev(6.0, 200.0)])
    (stats,) = rows.values()
    assert stats["samples"] == 2
    assert stats["latest"] == 6.0
    assert stats["avg"] == 5.0


def test_rejected_sample_still_counts_as_an_error_for_its_node() -> None:
    """Excluding it from the THROUGHPUT rollup must not hide it from the error
    accounting — the sample is a real event worth surfacing, just not a real
    measurement. This is why the fix is a narrow condition rather than skipping
    the event outright.
    """
    summary = gb_dataflux.summarize([
        _ev(5.0, 100.0),
        _ev(21065.2, 200.0, status="error"),
    ])
    assert summary["nodes"]["host"]["events"] == 2
    assert summary["nodes"]["host"]["errors"] == 1
    assert summary["by_kind"]["tok_s_measured"]["errors"] == 1


def test_unmarked_historical_outlier_is_left_alone() -> None:
    """Known limitation, asserted so it stays visible rather than being assumed
    fixed: samples written BEFORE the write-side ceiling existed carry no
    rejection marker, so there is nothing for the reader to honour. Cleaning
    those would need a read-side ceiling, which is a separate decision — the log
    is append-only and a blanket numeric cutoff would be guesswork about what
    any given node can actually do.
    """
    rows = _rollup([_ev(18355.3, 100.0)])  # status "ok" — pre-fix data
    (stats,) = rows.values()
    assert stats["latest"] == 18355.3
