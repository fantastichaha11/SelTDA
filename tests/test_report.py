from filtering.report import build_filter_report


def make_record(qid, score_conf, score_itm, score_xcons, keep_reason):
    return (
        {
            "question_id": qid,
            "question": f"q{qid}?",
            "answer": [f"a{qid}"],
            "image": f"vg/img_{qid}.jpg",
            "scores": {"conf": score_conf, "itm": score_itm, "xcons": score_xcons},
        },
        keep_reason,
    )


def test_report_counts():
    decisions = [
        make_record(0, 0.9, 0.9, 0.9, "kept"),
        make_record(1, 0.1, 0.9, 0.9, "conf"),
        make_record(2, 0.9, 0.1, 0.9, "itm"),
        make_record(3, 0.9, 0.9, 0.1, "xcons"),
        make_record(4, 0.9, 0.9, 0.9, "kept"),
    ]
    report = build_filter_report(decisions, max_examples=2)
    assert report["counts"]["total_in"] == 5
    assert report["counts"]["kept"] == 2
    assert report["counts"]["dropped"]["conf"] == 1
    assert report["counts"]["dropped"]["itm"] == 1
    assert report["counts"]["dropped"]["xcons"] == 1


def test_report_histograms_have_bins():
    decisions = [make_record(i, i / 10.0, i / 10.0, i / 10.0, "kept") for i in range(11)]
    report = build_filter_report(decisions, max_examples=0, n_bins=5)
    assert "histograms" in report
    assert len(report["histograms"]["conf"]) == 5


def test_report_includes_rejected_examples():
    decisions = [
        make_record(1, 0.1, 0.9, 0.9, "conf"),
        make_record(2, 0.2, 0.9, 0.9, "conf"),
        make_record(3, 0.9, 0.1, 0.9, "itm"),
    ]
    report = build_filter_report(decisions, max_examples=10)
    examples = report["rejected_examples"]
    assert {e["reason"] for e in examples} == {"conf", "itm"}
    assert all("question" in e and "answer" in e for e in examples)
