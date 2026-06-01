from orchestration.hrp import HrpState, respond


def test_l0_no_hacking_keeps_state():
    s = HrpState(kl_beta=0.1, w_itm=0.5)
    out, action = respond(
        s,
        hacking=False,
        dominant_term=None,
        judge_dropping=False,
        auto_levels=[1, 2, 3],
    )
    assert action == "continue"
    assert out.kl_beta == 0.1 and out.w_itm == 0.5


def test_l1_adaptive_kl_on_mild_hacking():
    s = HrpState(kl_beta=0.1, w_itm=0.5)
    out, action = respond(
        s,
        hacking=True,
        dominant_term=None,
        judge_dropping=False,
        auto_levels=[1, 2, 3],
        kl_c=1.5,
    )
    assert action == "adaptive_kl"
    assert abs(out.kl_beta - 0.15) < 1e-9


def test_l2_downweight_dominant_term():
    s = HrpState(kl_beta=0.1, w_itm=0.5)
    out, action = respond(
        s,
        hacking=True,
        dominant_term="itm",
        judge_dropping=False,
        auto_levels=[1, 2, 3],
        downweight_d=0.5,
    )
    assert action == "downweight"
    assert abs(out.w_itm - 0.25) < 1e-9


def test_l3_early_stop_when_judge_dropping():
    s = HrpState(kl_beta=0.1, w_itm=0.5)
    out, action = respond(
        s,
        hacking=True,
        dominant_term=None,
        judge_dropping=True,
        auto_levels=[1, 2, 3],
    )
    assert action == "early_stop_rollback"


def test_l4_flagged_for_human_when_not_in_auto_levels():
    s = HrpState(kl_beta=0.1, w_itm=0.5, persistent_rounds=2)
    out, action = respond(
        s,
        hacking=True,
        dominant_term=None,
        judge_dropping=True,
        auto_levels=[1, 2, 3],
    )
    assert action == "flag_human"
