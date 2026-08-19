"""Route triggers must be lowercase, or they can never fire.

gb_semantics.answer() lowercases the question and tests `trigger in question`.
A trigger carrying any capital letter is therefore dead on arrival , it looks
correct in the YAML and silently routes nothing. Found 2026-08-19 with a
"MemoryError" trigger that could not match by construction.
"""
import gb_semantics


def test_every_trigger_is_lowercase():
    dead = [(r.intent, t) for r in gb_semantics.load()["routes"]
            for t in r.triggers if t != t.lower()]
    assert not dead, ("triggers that can never match a lowercased question: "
                      + ", ".join(f"{i}:{t!r}" for i, t in dead))


def test_every_route_names_something_that_exists():
    L = gb_semantics.load()
    missing = []
    for r in L["routes"]:
        missing += [(r.intent, "metric", m) for m in r.metrics if m not in L["metrics"]]
        missing += [(r.intent, "segment", s) for s in r.segments if s not in L["segments"]]
    assert not missing, f"routes pointing at undefined names: {missing}"
