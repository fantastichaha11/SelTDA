import os
import subprocess
from pathlib import Path


def test_scripts_are_valid_bash_and_expose_all_conditions():
    for dataset in ("pathvqa", "vizwiz"):
        path = Path(f"scripts/run_{dataset}_judge_matrix.sh")
        completed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        text = path.read_text(encoding="utf-8")
        for condition in ("rubric_probe", "filter4", "pt4_j", "pt8_jp", "pt4_jp_60k"):
            assert condition in text
        assert "truncate_train_dataset_to" in text
        assert "max_epoch=10" in text
        assert "--no-resume" in text
        assert "compute.json" in text
        assert "commands.jsonl" in text


def test_pathvqa_filter4_dry_run_prints_primary_stages():
    env = {**os.environ, "DRY_RUN": "1"}
    completed = subprocess.run(
        ["bash", "scripts/run_pathvqa_judge_matrix.sh", "filter4"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "generate_questions.py" in completed.stdout
    assert "score_rank_prometheus_pool.py" in completed.stdout
    assert "train_vqa.py" in completed.stdout
    assert "pathvqa_eval.py" in completed.stdout


def test_vizwiz_60k_dry_run_skips_teacher_training():
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "PT4_JP_SYNTHETIC_40K": "datasets/vizwiz/synthetic_full_40k.json",
    }
    completed = subprocess.run(
        ["bash", "scripts/run_vizwiz_judge_matrix.sh", "pt4_jp_60k"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "build_student_synthetic.py" in completed.stdout
    assert "train_vqa.py" in completed.stdout
    assert "vizwiz_eval.py" in completed.stdout
    assert "train_teacher_grpo.py" not in completed.stdout
