"""tok_s_decode is the sustained rate, not the latest burst."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_semantics as sem


def _write(log, rows):
    now = time.time()
    with open(log, "w") as f:
        for i, (tok_s, ctok) in enumerate(rows):
            f.write(json.dumps({"ts": now - (len(rows) - i), "kind": "tok_s_measured", "status": "ok", "node": "host", "label": "gb_synapse",
                                "n_items": 1, "items": ["m"], "duration_s": 0.0, "model": "m", "tok_s": tok_s, "completion_tokens": ctok}) + "\n")


def test_a_short_burst_does_not_define_the_decode_rate(tmp_path, monkeypatch):
    log = tmp_path / "df.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(log))
    # the real shape seen 2026-10-30: replies of 32-41 tokens report 93-318 tok/s, the 399-token reply 58
    _write(log, [(58.1, 399), (131.2, 36), (93.9, 69), (60.2, 250), (318.6, 32)])
    r = sem._res_tok_s_decode(None, None)
    assert r["value"] is not None and 55 < r["value"] < 65, r             # token-weighted over the two long replies
    assert r["value"] != 318.6 and "token-weighted" in r["raw_source"]
    assert r["samples"] == 2


def test_with_no_long_reply_the_median_is_used_and_it_says_bursts_are_possible(tmp_path, monkeypatch):
    log = tmp_path / "df.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(log))
    _write(log, [(90.0, 36), (130.0, 34), (318.6, 32)])
    r = sem._res_tok_s_decode(None, None)
    assert r["value"] == 130.0 and "short-reply bursts" in r["raw_source"]
