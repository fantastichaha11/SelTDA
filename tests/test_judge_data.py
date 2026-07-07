import json

from judge.data import (
    JudgePair,
    NegativeSampler,
    build_answer_pools,
    build_judge_pairs,
    infer_answer_type,
    load_image_pool,
    question_prefix,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_question_prefix_and_answer_type():
    assert question_prefix("How many nuclei are visible?") == "how many"
    assert question_prefix("Is the tissue abnormal?") == "is/are"
    assert question_prefix("What stain is shown?") == "what"
    assert infer_answer_type("yes") == "yes/no"
    assert infer_answer_type("12") == "count"
    assert infer_answer_type("squamous cell carcinoma") == "phrase"


def test_answer_pools_use_only_supplied_train_records():
    train = [
        {"image": "img/a.jpg", "question": "Is this benign?", "answer": "yes"},
        {"image": "img/b.jpg", "question": "What is visible?", "answer": "nuclei"},
    ]

    pools = build_answer_pools(train)

    assert pools.all_answers == ["nuclei", "yes"]
    assert pools.by_prefix["is/are"] == ["yes"]
    assert pools.by_type["phrase"] == ["nuclei"]


def test_negative_sampler_flips_yes_no_before_pool_sampling():
    train = [{"image": "img/a.jpg", "question": "Is this benign?", "answer": "yes"}]

    sampler = NegativeSampler(build_answer_pools(train), seed=7)

    assert sampler.sample(train[0]) == "no"


def test_negative_sampler_uses_train_pool_for_val_record():
    train = [
        {"image": "img/a.jpg", "question": "What is visible?", "answer": "nuclei"},
        {"image": "img/b.jpg", "question": "What is present?", "answer": "stroma"},
    ]
    val_record = {
        "image": "img/c.jpg",
        "question": "What is highlighted?",
        "answer": "val-only-answer",
    }

    sampler = NegativeSampler(build_answer_pools(train), seed=3)

    assert sampler.sample(val_record) in {"nuclei", "stroma"}


def test_build_judge_pairs_emits_positive_and_negative_candidates():
    train = [
        {"image": "img/a.jpg", "question": "Is this benign?", "answer": "yes"},
        {"image": "img/b.jpg", "question": "What is visible?", "answer": "nuclei"},
    ]

    pairs = build_judge_pairs(
        records=train,
        pools=build_answer_pools(train),
        seed=11,
        negatives_per_positive=1,
    )

    assert len(pairs) == 2
    assert all(isinstance(pair, JudgePair) for pair in pairs)
    assert pairs[0].positive.label == 1
    assert pairs[0].negative.label == 0
    assert pairs[0].negative.answer == "no"


def test_load_image_pool_deduplicates_images_and_strips_gt_qa_when_disabled(tmp_path):
    ann = tmp_path / "train.json"
    _write_json(
        ann,
        [
            {"image": "img/a.jpg", "question": "Q1?", "answer": "yes"},
            {"image": "img/a.jpg", "question": "Q2?", "answer": "no"},
            {"image": "img/b.jpg", "question": "Q3?", "answer": "nuclei"},
        ],
    )

    pool = load_image_pool(
        {
            "name": "pathvqa_train",
            "annotations": str(ann),
            "image_root": str(tmp_path / "images"),
            "use_ground_truth_qa": False,
        }
    )

    assert [item.image for item in pool] == ["img/a.jpg", "img/b.jpg"]
    assert all(item.question is None and item.answer is None for item in pool)
