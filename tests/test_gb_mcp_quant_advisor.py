"""Tests for gb_mcp.quant_advisor()'s per_tensor_plan extension
(missing_features.md item (j)) — the MCP surface for gb_gguf_plan.py, added
per this repo's MCP-Tool-Gaps rule (a new capability must be reachable live
over MCP, not only from a script).
"""
import gb_mcp


class _FakeReport:
    def __init__(self, name="qwen35", quant="Q4_K_M"):
        self.name = name
        self.quant = quant
        self.total_gb = 18.0
        self.ctx = 65536
        self.fits_vram = True
        self.est_tok_s = 5.27
        self.measured = True
        self.note = ""


def test_quant_advisor_default_omits_per_tensor_plan_key(monkeypatch):
    monkeypatch.setattr("gb_synapse.recommend", lambda ctx=65536, probe_feeders=True: [_FakeReport()])

    result = gb_mcp.quant_advisor()

    assert "per_tensor_plan" not in result


def test_quant_advisor_per_tensor_plan_requires_source_path(monkeypatch):
    monkeypatch.setattr("gb_synapse.recommend", lambda ctx=65536, probe_feeders=True: [_FakeReport()])

    result = gb_mcp.quant_advisor(per_tensor_plan=True)

    assert "error" in result["per_tensor_plan"]
    assert "source_gguf_path" in result["per_tensor_plan"]["error"]


def test_quant_advisor_per_tensor_plan_computes_plan(monkeypatch):
    monkeypatch.setattr("gb_synapse.recommend", lambda ctx=65536, probe_feeders=True: [_FakeReport()])
    monkeypatch.setattr("gb_gguf_plan.read_gguf_tensor_inventory",
                        lambda path: [("blk.0.ffn_gate.weight", 4 * 2 ** 30),
                                      ("blk.0.ssm_out.weight", 1 * 2 ** 30)])
    monkeypatch.setattr("gb_synapse_backends.effective_vram_budget_mb",
                        lambda: (3000.0, 4000.0, {"t2_free_mb": 2000.0}))

    result = gb_mcp.quant_advisor(per_tensor_plan=True, source_gguf_path="/fake/q8_0.gguf")

    plan = result["per_tensor_plan"]
    assert plan["source_gguf_path"] == "/fake/q8_0.gguf"
    assert plan["budget_mb"] == 4000.0
    assert "ssm" in plan["role_breakdown"]
    assert plan["role_breakdown"]["ssm"]["floor_type"] == "q8_0"
    assert "n_tensor_overrides" in plan


def test_quant_advisor_per_tensor_plan_reports_read_error(monkeypatch):
    monkeypatch.setattr("gb_synapse.recommend", lambda ctx=65536, probe_feeders=True: [_FakeReport()])

    def _boom(path):
        raise RuntimeError("not a gguf file")
    monkeypatch.setattr("gb_gguf_plan.read_gguf_tensor_inventory", _boom)

    result = gb_mcp.quant_advisor(per_tensor_plan=True, source_gguf_path="/bad/path.gguf")

    assert "error" in result["per_tensor_plan"]
    assert "not a gguf file" in result["per_tensor_plan"]["error"]
