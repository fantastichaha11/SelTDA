from filtering.audit import detect_hacking, judge_score


def test_judge_score_is_mean_of_components():
    assert abs(judge_score(itm_large=0.8, vqa_match=0.6) - 0.7) < 1e-9


def test_detect_hacking_reward_up_judge_down():
    assert (
        detect_hacking(
            d_reward=0.10,
            d_judge=-0.05,
            eps=0.02,
            term_share={"itm": 0.3},
            tau=0.6,
        )
        is True
    )


def test_detect_hacking_term_dominance():
    assert (
        detect_hacking(
            d_reward=0.10,
            d_judge=0.01,
            eps=0.02,
            term_share={"itm": 0.7},
            tau=0.6,
        )
        is True
    )


def test_no_hacking_when_judge_rises():
    assert (
        detect_hacking(
            d_reward=0.10,
            d_judge=0.05,
            eps=0.02,
            term_share={"itm": 0.3},
            tau=0.6,
        )
        is False
    )
