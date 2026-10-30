"""gb_dataflux.spill_outcome — classify a tier move against an open spill.

Recorded at the emit site rather than inferred later, because after the fact
"demoted and still resident in T2" and "demoted and then evicted" are the same
observation: no promotion followed. Only the mover knows which happened.
"""
import gb_dataflux as d
import pytest

T1, T2, T3 = "T1_HBM", "T2_DDR", "T3_NVME"


@pytest.fixture(autouse=True)
def _clean():
    d.reset_spill_state()
    yield
    d.reset_spill_state()


def test_demotion_opens_a_spill():
    assert d.spill_outcome("unet", T1, T2, 8.0, now=100, pid=1) == {"spill_outcome": "opened"}


def test_promotion_closes_it_as_returned():
    d.spill_outcome("unet", T1, T2, 8.0, now=100, pid=1)
    r = d.spill_outcome("unet", T2, T1, 8.0, now=125, pid=1)
    assert r["spill_outcome"] == "returned"
    assert r["spill_resident_s"] == 25.0
    assert r["spill_returns"] == 1


def test_pushing_further_down_closes_it_as_dropped():
    """T2 -> T3 is upstream's host-dropped case: it will not be undone from here."""
    d.spill_outcome("vae", T1, T2, 2.0, now=100, pid=1)
    r = d.spill_outcome("vae", T2, T3, 2.0, now=180, pid=1)
    assert r["spill_outcome"] == "dropped"
    assert r["spill_resident_s"] == 80.0


def test_promotion_without_an_open_spill_says_nothing():
    assert d.spill_outcome("unet", T2, T1, 8.0, now=100, pid=1) == {}


def test_returns_accumulate_across_round_trips():
    for i in range(3):
        d.spill_outcome("unet", T1, T2, 8.0, now=100 + 10 * i, pid=1)
        r = d.spill_outcome("unet", T2, T1, 8.0, now=105 + 10 * i, pid=1)
    assert r["spill_returns"] == 3


def test_state_is_keyed_per_process():
    """Two processes can hold same-named components; one's promotion must not
    close the other's spill."""
    d.spill_outcome("unet", T1, T2, 8.0, now=100, pid=1)
    assert d.spill_outcome("unet", T2, T1, 8.0, now=110, pid=2) == {}
    assert d.spill_outcome("unet", T2, T1, 8.0, now=120, pid=1)["spill_outcome"] == "returned"


def test_never_raises_on_bad_input():
    """Observation must not break a tier move."""
    assert d.spill_outcome(None, None, None, None) == {}


def test_fields_are_registered_in_the_schema():
    import gb_dataflux_kinds as k
    spec = k.KINDS["tier_move"]
    assert "spill_outcome" in spec.fields
    assert "spill_resident_s" in spec.numeric_fields
    assert "spill_returns" in spec.numeric_fields
