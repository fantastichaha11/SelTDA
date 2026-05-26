from filtering.gate_registry import GATE_ORDER, enabled_gate_names


def test_gate_order_includes_grounding_gates():
    assert GATE_ORDER.index("lp") < GATE_ORDER.index("kcons")


def test_enabled_gate_names_with_lp_kcons():
    class G:
        enabled = True
        keep_top = 0.75

    class Gates:
        conf = G()
        itm = G()
        xcons = G()
        lp = G()
        kcons = type("K", (), {"enabled": False, "keep_top": 0.75})()

    class Config:
        gates = Gates()

    names = enabled_gate_names(Config())
    assert names == ["conf", "itm", "xcons", "lp"]
