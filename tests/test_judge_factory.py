from judge.factory import resolve_prometheus_config
from judge.prometheus import DEFAULT_RUBRIC


def test_resolve_uses_default_only_when_rubric_is_absent(tmp_path):
    path = tmp_path / "judge.yaml"
    path.write_text("model:\n  model_path: base\n  model_base: null\n", encoding="utf-8")

    resolved = resolve_prometheus_config(path)

    assert resolved.rubric == DEFAULT_RUBRIC
    assert resolved.model_base is None


def test_resolve_preserves_embedded_rubric_and_name(tmp_path):
    path = tmp_path / "judge.yaml"
    path.write_text(
        "model:\n"
        "  model_path: base\n"
        "prompt:\n"
        "  rubric_name: qa8\n"
        "  criteria_count: 8\n"
        "  rubric: eight criteria\n",
        encoding="utf-8",
    )

    resolved = resolve_prometheus_config(
        path, selected_model_path="adapter", device="cuda:1"
    )

    assert resolved.model_path == "adapter"
    assert resolved.device == "cuda:1"
    assert resolved.rubric_name == "qa8"
    assert resolved.criteria_count == 8
    assert resolved.rubric == "eight criteria"
