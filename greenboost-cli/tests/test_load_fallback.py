"""A load that dies of OOM is retried lighter; other stops are left alone."""
import os
import time

from greenboost_cli.inference import load_fallback as LF

OOM = "[GreenBoost] OOM in smart_alloc: 304 MiB requested\nterminate called\n  what():  failed to allocate compute pp buffers\n"


def _run(tmp_path, statuses, log_text, ladder=None):
    log = tmp_path / "m.log"
    if log_text is not None:
        log.write_text(log_text)
    seq = iter(statuses)
    said, relaunched = [], []
    out = LF.watch_and_retry(status_fn=lambda: next(seq, "running"), log_file=log,
                             relaunch=lambda ub, moe: relaunched.append((ub, moe)),
                             say=said.append, sleep=lambda s: None, ladder=ladder)
    return out, said, relaunched


def test_oom_death_retries_with_the_first_lighter_rung(tmp_path):
    out, said, relaunched = _run(tmp_path, ["starting", "stopped", "starting", "running"], OOM)
    assert out == "running" and relaunched == [(1024, None)]
    assert "ran out of VRAM" in said[0] and "micro-batch 1024" in said[0]


def test_keeps_stepping_down_then_gives_up_with_advice(tmp_path):
    out, said, relaunched = _run(tmp_path, ["stopped"] * 10, OOM, ladder=[(1024, None), (512, 40)])
    assert out == "exhausted" and relaunched == [(1024, None), (512, 40)]
    assert "/gpu" in said[-1] and "nothing is killed" in said[-1]


def test_a_stop_that_is_not_oom_is_not_retried(tmp_path):
    out, _, relaunched = _run(tmp_path, ["starting", "stopped"], "all fine, stopped by user\n")
    assert out == "gone" and relaunched == []


def test_stale_log_from_an_earlier_launch_is_ignored(tmp_path):
    log = tmp_path / "m.log"
    log.write_text(OOM)
    os.utime(log, (time.time() - 3600, time.time() - 3600))
    seq = iter(["stopped"])
    relaunched = []
    out = LF.watch_and_retry(status_fn=lambda: next(seq), log_file=log, relaunch=lambda *a: relaunched.append(a),
                             say=lambda m: None, sleep=lambda s: None)
    assert out == "gone" and relaunched == []


def test_signatures():
    assert LF.log_shows_oom(OOM) and not LF.log_shows_oom("ready")
    assert LF.describe((512, 40)) == "micro-batch 512, 40 expert layers on the CPU"
