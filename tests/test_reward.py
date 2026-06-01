from filtering.reward import repetition_penalty, type_match


def test_type_match_in_weak_set():
    assert type_match("How many dogs?", "3", {"how_many"}) == 1.0


def test_type_match_not_in_weak_set():
    assert type_match("Is it red?", "yes", {"how_many"}) == 0.0


def test_repetition_penalty_unique_is_zero():
    seen = ["how many dogs are there", "what color is the car"]
    assert repetition_penalty("is the sky blue", seen, threshold=0.9) == 0.0


def test_repetition_penalty_duplicate_is_one():
    seen = ["how many dogs are there"]
    assert repetition_penalty("how many dogs are there", seen, threshold=0.9) == 1.0
