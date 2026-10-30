"""One definition of "how many layers hold a KV cache".

The rule had five copies: the `n_layers - n_recurrent_layers` subtraction twice
during GGUF parsing (that half is the source), and the fallback rule three
times — twice in gb_synapse and once in gb_semantics — each carrying its own
near-verbatim copy of the same warning comment. The reasoning was re-derived
every time because there was nothing to call.

That is precisely how casebook C-07 happened: every caller of estimate_kv_gb was
corrected to charge n_kv_layers except one, and gb_mcp.py kept overestimating by
3.58x (5.85 GiB of phantom reservation on an 11.3 GB card) until 2026-08-22. A
rule in five places is a rule that will be fixed in four.
"""
import pytest

import gb_synapse


class _Entry:
    def __init__(self, **kw):
        self.is_recurrent_only = False
        self.n_kv_layers = 0
        self.n_layers = 0
        self.__dict__.update(kw)


def test_hybrid_charges_only_the_attention_layers():
    """The served model: 65 blocks, 17 attention + 48 Gated DeltaNet."""
    e = _Entry(n_kv_layers=17, n_layers=65)
    assert gb_synapse.effective_kv_layers(e) == 17


def test_recurrent_only_is_zero_not_missing():
    """THE trap. `or n_layers` here is the ~4x overestimate that clamped ctx
    far below what the hardware holds."""
    e = _Entry(is_recurrent_only=True, n_kv_layers=0, n_layers=64)
    assert gb_synapse.effective_kv_layers(e) == 0


def test_unknown_geometry_falls_back_to_every_layer():
    """A dense model with no split recorded: every layer really does hold KV."""
    e = _Entry(n_kv_layers=0, n_layers=32)
    assert gb_synapse.effective_kv_layers(e) == 32


def test_explicit_n_layers_overrides_the_entry():
    e = _Entry(n_kv_layers=0, n_layers=32)
    assert gb_synapse.effective_kv_layers(e, 40) == 40


def test_recurrent_only_ignores_an_explicit_n_layers_too():
    e = _Entry(is_recurrent_only=True, n_kv_layers=0, n_layers=64)
    assert gb_synapse.effective_kv_layers(e, 99) == 0


def test_missing_attributes_do_not_raise():
    """Entries come from several places; none of them may crash sizing."""
    class Bare:
        pass
    assert gb_synapse.effective_kv_layers(Bare()) == 0


def test_the_rule_has_exactly_one_implementation():
    """Guard against the copies coming back."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    pattern = "is_recurrent_only else ("
    hits = []
    for f in ("gb_synapse.py", "gb_semantics.py", "gb_mcp.py", "gb_synapse_backends.py"):
        p = repo / f
        if p.exists() and pattern in p.read_text():
            hits.append(f)
    assert not hits, f"inline copies of the KV-layer rule reappeared in: {hits}"
