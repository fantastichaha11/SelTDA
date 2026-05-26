from orchestration.merge_pools import merge_with_weights


def test_merge_duplicates_by_weight():
    easy = [
        {
            "question_id": 1,
            "question": "Q?",
            "answer": ["a"],
            "image": "x.jpg",
            "dataset": "aokvqa",
        }
    ]
    hard = [
        {
            "question_id": 2,
            "question": "Q2?",
            "answer": ["b"],
            "image": "y.jpg",
            "dataset": "aokvqa",
        }
    ]
    merged = merge_with_weights({"easy": (easy, 1), "hard": (hard, 3)})
    assert len(merged) == 4
    assert sum(1 for r in merged if r["question_id"] == 2) == 3
