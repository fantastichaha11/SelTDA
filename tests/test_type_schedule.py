from orchestration.type_schedule import next_question_type, type_prompt_prefix


def test_round_robin():
    assert next_question_type(0, "round_robin") == "yes_no"
    assert next_question_type(1, "round_robin") == "how_many"


def test_target_weak():
    weak = ["external_knowledge", "yes_no"]
    t = next_question_type(0, "target_weak", weak_types=weak)
    assert t in weak or t == "yes_no"


def test_type_prompt_prefix():
    assert type_prompt_prefix("how_many") == "[TYPE=how_many] "
