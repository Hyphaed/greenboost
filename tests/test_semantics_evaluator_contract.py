"""Every segment evaluator must honour the (matched, evidence) tuple contract.

`evaluate_segment()` does `matched, evidence = fn()`. An evaluator that returns
a dict instead unpacks into its two KEY NAMES and then fails, or worse succeeds
with nonsense. That shipped once: `_seg_models_wiped_from_manifest` was written
returning a dict, passed the coverage gate and its own structural eval (both of
which only check that the function runs), and only broke when something called
it through the real public API.

`matched` is three-valued on purpose: True, False, or None for "cannot tell".
A segment that cannot determine its verdict must return None, never False —
a clean bill inferred from missing data is the failure this layer prevents.
"""
import gb_semantics as S


def _segment_names():
    return sorted(S.load()["segments"].keys())


def test_there_are_segments_to_check():
    assert _segment_names(), "no segments loaded — the rest of this file would vacuously pass"


def test_every_evaluator_returns_the_tuple_contract():
    bad = []
    for name in _segment_names():
        seg = S.load()["segments"][name]
        fn = S.__dict__.get("_seg_" + seg.evaluator)
        if fn is None:
            bad.append(f"{name}: evaluator _seg_{seg.evaluator} missing")
            continue
        try:
            out = fn()
        except Exception as e:
            bad.append(f"{name}: raised {type(e).__name__}: {e}")
            continue
        if not isinstance(out, tuple) or len(out) != 2:
            bad.append(f"{name}: returned {type(out).__name__}, expected a 2-tuple")
            continue
        matched, evidence = out
        if matched not in (True, False, None):
            bad.append(f"{name}: matched={matched!r}, expected True/False/None")
        if not isinstance(evidence, (list, tuple)):
            bad.append(f"{name}: evidence is {type(evidence).__name__}, expected a list")
    assert not bad, "evaluator contract violations:\n  " + "\n  ".join(bad)


def test_every_segment_evaluates_through_the_public_api():
    """The path a real caller takes, which the structural evals do not exercise."""
    bad = []
    for name in _segment_names():
        r = S.evaluate_segment(name)
        if r.get("error"):
            bad.append(f"{name}: {r['error']}")
        elif r.get("matched") not in (True, False, None):
            bad.append(f"{name}: matched={r.get('matched')!r}")
    assert not bad, "evaluate_segment() failures:\n  " + "\n  ".join(bad)
