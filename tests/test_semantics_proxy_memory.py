"""Proxy-memory metrics degrade to "unknown", never to "fine".

Context (2026-08-19): an unbounded read_events() memo grew the gb-synapse
proxy at ~10.7 MB/s while idle until it hit its RLIMIT_AS cap and raised
MemoryError mid-generation. Host memory metrics looked perfect throughout ,
32 GB available , because the cap is per-process. These metrics exist to make
that visible, so their failure modes matter as much as their happy path.
"""
import gb_semantics


def test_absent_proxy_resolves_to_none_with_a_reason(monkeypatch):
    monkeypatch.setattr(gb_semantics, "_proxy_mem",
                        lambda: (None, None, "no gb_synapse_api process is running"))
    for name in ("proxy_rss_mb", "proxy_mem_headroom_pct"):
        r = gb_semantics.resolve(name)
        assert r["value"] is None
        assert "no gb_synapse_api process" in r["provenance"]["raw_source"]


def test_absent_proxy_segment_is_none_not_false(monkeypatch):
    """A clean bill of health inferred from absent data is the failure mode
    the governed layer exists to prevent."""
    monkeypatch.setattr(gb_semantics, "_proxy_mem",
                        lambda: (None, None, "no gb_synapse_api process is running"))
    out = gb_semantics.evaluate_segment("proxy_memory_near_cap")
    assert out["matched"] is None


def test_uncapped_proxy_is_none_not_full_headroom(monkeypatch):
    """No cap means the question is unanswerable, not that headroom is 100%."""
    monkeypatch.setattr(gb_semantics, "_proxy_mem", lambda: (5000.0, None, "/proc/1/x"))
    assert gb_semantics.resolve("proxy_mem_headroom_pct")["value"] is None
    assert gb_semantics.evaluate_segment("proxy_memory_near_cap")["matched"] is None


def test_near_cap_matches(monkeypatch):
    """5441 MB against the real 6.15 GiB cap , the live 2026-08-19 numbers."""
    monkeypatch.setattr(gb_semantics, "_proxy_mem",
                        lambda: (5441.0, 6298.0, "/proc/838350/{status,limits}"))
    out = gb_semantics.evaluate_segment("proxy_memory_near_cap")
    assert out["matched"] is True
    assert gb_semantics.resolve("proxy_mem_headroom_pct")["value"] < 20.0


def test_healthy_proxy_does_not_match(monkeypatch):
    monkeypatch.setattr(gb_semantics, "_proxy_mem",
                        lambda: (400.0, 6298.0, "/proc/1/{status,limits}"))
    assert gb_semantics.evaluate_segment("proxy_memory_near_cap")["matched"] is False


def test_a_shell_mentioning_the_proxy_is_not_the_proxy(tmp_path, monkeypatch):
    """Regression: matching the joined cmdline matched the grep looking for it.

    /proc/<pid>/cmdline is NUL-separated. A substring test over the joined
    string matched any shell whose command line merely CONTAINED the name ,
    the first version of this resolver reported 7.6 MB of "proxy" that was
    really a pipeline inspecting it.
    """
    import glob as real_glob

    proc = tmp_path / "proc"
    # a shell that merely mentions the proxy, as one long argv token
    imposter = proc / "111"
    imposter.mkdir(parents=True)
    (imposter / "cmdline").write_bytes(b"/bin/bash\x00-c\x00pgrep -af gb_synapse_api.py\x00")
    (imposter / "status").write_text("VmRSS:\t   7600 kB\n")
    (imposter / "limits").write_text("Max address space         unlimited            unlimited            bytes\n")
    # the real thing: the script is its own argv token
    real = proc / "222"
    real.mkdir()
    (real / "cmdline").write_bytes(b"/usr/bin/python3\x00/usr/local/lib/greenboost/gb_synapse_api.py\x00")
    (real / "status").write_text("VmRSS:\t 5572000 kB\n")
    (real / "limits").write_text("Max address space         6604117196           6604117196           bytes\n")

    monkeypatch.setattr(real_glob, "glob",
                        lambda pat: [str(p / "cmdline") for p in (imposter, real)]
                        if "cmdline" in pat else [])

    rss, cap, prov = gb_semantics._proxy_mem()
    assert rss is not None and rss > 5000      # picked the real proxy
    assert cap is not None                      # not the "unlimited" imposter
