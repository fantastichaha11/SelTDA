from orchestration.weak_types import error_rate_per_type, top_k_weak_types

RESULTS = [
    {"question": "How many cats?", "answer": "2", "pred": "3"},
    {"question": "How many cats?", "answer": "2", "pred": "2"},
    {"question": "What sport is this?", "answer": "golf", "pred": "tennis"},
    {"question": "Is it blue?", "answer": "yes", "pred": "yes"},
]


def test_error_rate_per_type():
    er = error_rate_per_type(RESULTS)
    assert abs(er["how_many"] - 0.5) < 1e-9
    assert abs(er["external_knowledge"] - 1.0) < 1e-9
    assert abs(er["yes_no"] - 0.0) < 1e-9


def test_top_k_weak_types():
    assert top_k_weak_types(RESULTS, k=2) == ["external_knowledge", "how_many"]
