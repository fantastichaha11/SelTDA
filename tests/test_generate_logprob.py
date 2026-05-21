from generate_questions import VQARecord


def test_vqa_record_accepts_gen_logprob():
    r = VQARecord(
        question_id=0,
        question="q?",
        answer=["a"],
        image="vg/img.jpg",
        dataset="vg",
        gen_logprob=-1.23,
    )
    assert r.gen_logprob == -1.23


def test_vqa_record_default_gen_logprob_is_none():
    r = VQARecord(
        question_id=0,
        question="q?",
        answer=["a"],
        image="vg/img.jpg",
        dataset="vg",
    )
    assert r.gen_logprob is None
    assert r.scores is None
