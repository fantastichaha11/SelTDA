from filtering.strata import classify_question_type, thresholds_per_stratum


def test_classify_yes_no():
    assert classify_question_type("Is the dog brown?", "yes") == "yes_no"


def test_classify_how_many():
    assert classify_question_type("How many people are there?", "3") == "how_many"


def test_classify_ek():
    q = "What kind of sport is this?"
    assert classify_question_type(q, "soccer") == "external_knowledge"


def test_thresholds_per_stratum_separate_quantiles():
    records = [
        {"question": "Is it?", "answer": ["yes"], "scores": {"conf": 0.9}},
        {"question": "Is it blue?", "answer": ["no"], "scores": {"conf": 0.1}},
        {"question": "What is shown?", "answer": ["cat"], "scores": {"conf": 0.5}},
        {"question": "What object?", "answer": ["dog"], "scores": {"conf": 0.6}},
    ]
    tau = thresholds_per_stratum(records, gate="conf", keep_top=0.5, min_stratum_size=1)
    assert "yes_no" in tau
    assert "visual_reasoning" in tau
    assert tau["yes_no"] != tau["visual_reasoning"]
