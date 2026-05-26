from unittest.mock import MagicMock

from orchestration.iterative_seltda import run_iterative_loop


def test_iterative_loop_early_stop(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, dry_run=False):
        calls.append(cmd[1] if len(cmd) > 1 else cmd[0])
        return 0

    results_r0 = tmp_path / "results_round_0.json"
    results_r1 = tmp_path / "results_round_1.json"
    results_r0.write_text(
        '[{"question": "Is it?", "answer": "yes", "correct": true},'
        '{"question": "How many?", "answer": "2", "correct": true}]'
    )
    results_r1.write_text(
        '[{"question": "Is it?", "answer": "yes", "correct": true},'
        '{"question": "How many?", "answer": "2", "correct": false}]'
    )

    config = {
        "project_root": str(tmp_path),
        "max_rounds": 2,
        "early_stop_delta": 0.5,
        "generate_config": "configs/generate_questions_aokvqa.yaml",
        "filter_config": "configs/filter_pseudo_stratified.yaml",
        "train_vqa_config": "configs/aokvqa.yaml",
        "eval_config": "configs/eval.yaml",
        "synthetic_path": str(tmp_path / "synthetic.json"),
        "results_template": str(tmp_path / "results_round_{round}.json"),
        "skill_gap_template": str(tmp_path / "skill_gap_{round}.json"),
        "staged_curriculum": False,
    }

    states = run_iterative_loop(config, subprocess_runner=fake_run)
    assert len(states) >= 1
    assert any("generate_questions" in c for c in calls)
