from filtering.gate_registry import GATE_SCORERS, apply_cascade, enabled_gate_names


def test_enabled_gate_names_respects_config():
    class G:
        enabled = True
        keep_top = 0.75

    class Gates:
        conf = G()
        itm = G()
        vqascore = type("V", (), {"enabled": False, "keep_top": 0.75})()
        xcons = type("X", (), {"enabled": False, "keep_top": 0.75})()

    class Config:
        gates = Gates()

    assert enabled_gate_names(Config()) == ["conf", "itm"]


def test_enabled_gate_names_includes_vqascore_when_enabled():
    class G:
        enabled = True
        keep_top = 0.75

    class Gates:
        conf = type("C", (), {"enabled": False, "keep_top": 0.75})()
        itm = type("I", (), {"enabled": False, "keep_top": 0.75})()
        vqascore = G()
        xcons = type("X", (), {"enabled": False, "keep_top": 0.75})()

    class Config:
        gates = Gates()

    assert enabled_gate_names(Config()) == ["vqascore"]


def test_apply_cascade_fails_at_first_gate():
    scores = {"conf": 0.2, "itm": 0.9, "xcons": 0.9}
    thresholds = {"conf": 0.5, "itm": 0.5, "xcons": 0.5}
    keep, reason = apply_cascade(scores, thresholds, ["conf", "itm", "xcons"])
    assert keep is False
    assert reason == "conf"


def test_apply_cascade_keeps_when_all_pass():
    scores = {"conf": 0.9, "itm": 0.9, "xcons": 0.9}
    thresholds = {"conf": 0.5, "itm": 0.5, "xcons": 0.5}
    keep, reason = apply_cascade(scores, thresholds, ["conf", "itm", "xcons"])
    assert keep is True
    assert reason == "kept"
