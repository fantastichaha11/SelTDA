import pytest

from judge.prometheus import (
    NO_REFERENCE_TEXT,
    PrometheusVisionScorer,
    StaticPrometheusScorer,
    _feedback_from_raw,
    build_prometheus_prompt,
    parse_prometheus_score,
    score_to_reward,
)


def test_build_prometheus_prompt_uses_no_reference_template():
    prompt = build_prometheus_prompt(
        question="What abnormality is visible?",
        candidate_answer="A necrotic tumor is present.",
    )

    assert "###Task Description:" in prompt
    assert (
        "An instruction, a response to evaluate, an image, a score rubric, and a neutral no-reference field are given."
        in prompt
    )
    assert "1. Write detailed feedback that assesses the response strictly based on the score rubric." in prompt
    assert "2. After writing feedback, write a score that is an integer between 1 and 5." in prompt
    assert (
        "3. The output format should be: Feedback: (feedback) [RESULT] (integer number between 1 and 5)"
        in prompt
    )
    assert "4. Do not generate any other opening, closing, or explanation." in prompt
    assert "###The instruction to evaluate:" in prompt
    assert "###Response to evaluate:" in prompt
    assert "###Reference Answer (Score 5):" in prompt
    assert "###Score Rubrics:" in prompt
    assert NO_REFERENCE_TEXT in prompt
    assert "Answer the visual question about this pathology image:\nWhat abnormality is visible?" in prompt
    assert "A necrotic tumor is present." in prompt
    assert "ground truth" not in prompt.lower()
    assert prompt.endswith("###Feedback:")


def test_parse_prometheus_score_parses_result_tag():
    assert parse_prometheus_score("Feedback: The answer is grounded. [RESULT] 4") == 4


def test_parse_prometheus_score_parses_score_label():
    assert parse_prometheus_score("Feedback: too generic. Score: 2") == 2


def test_parse_prometheus_score_parses_score_is_form():
    assert parse_prometheus_score("Feedback: grounded, score is 5") == 5


def test_parse_prometheus_score_raises_when_missing():
    with pytest.raises(ValueError):
        parse_prometheus_score("Feedback only. No numeric result provided.")


@pytest.mark.parametrize(
    ("raw_text", "feedback"),
    [
        ("Feedback: too generic. [RESULT] 2", "too generic."),
        ("Feedback: too generic. Score: 2", "too generic."),
        ("Feedback: too generic, score is 5", "too generic,"),
    ],
)
def test_feedback_from_raw_strips_all_supported_score_suffixes(raw_text, feedback):
    assert _feedback_from_raw(raw_text) == feedback


@pytest.mark.parametrize(
    ("score", "reward"),
    [
        (1, 0.0),
        (3, 0.5),
        (5, 1.0),
    ],
)
def test_score_to_reward_maps_linearly(score, reward):
    assert score_to_reward(score) == reward


def test_static_prometheus_scorer_returns_deterministic_result():
    result = StaticPrometheusScorer(score=4, feedback="ok").score(
        image_path="/tmp/slide.png",
        question="What is shown?",
        candidate_answer="Tumor cells.",
    )

    assert result.score == 4
    assert result.reward == 0.75
    assert result.feedback == "ok"
    assert result.raw_text == "Feedback: ok [RESULT] 4"


def test_prometheus_vision_scorer_generate_decodes_only_continuation(monkeypatch):
    scorer = PrometheusVisionScorer(model_path="stub")

    class FakeTensor:
        def __init__(self, data):
            self.data = data

        @property
        def shape(self):
            if self.data and isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)

        def unsqueeze(self, dim):
            assert dim == 0
            return FakeTensor([self.data])

        def to(self, *args, **kwargs):
            return self

        def tolist(self):
            return self.data

        def __getitem__(self, key):
            if isinstance(key, tuple):
                row_key, col_key = key
                rows = self.data[row_key]
                if not isinstance(rows, list) or (rows and not isinstance(rows[0], list)):
                    rows = [rows]
                return FakeTensor([row[col_key] for row in rows])
            return self.data[key]

    class FakeImage:
        def convert(self, mode):
            assert mode == "RGB"
            return self

    class FakeImageModule:
        @staticmethod
        def open(path):
            assert path == "/tmp/slide.png"
            return FakeImage()

    class FakeInferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTorch:
        float16 = "float16"

        @staticmethod
        def inference_mode():
            return FakeInferenceMode()

    class FakeConv:
        roles = ("user", "assistant")

        def __init__(self):
            self.messages = []

        def copy(self):
            return FakeConv()

        def append_message(self, role, content):
            self.messages.append((role, content))

        def get_prompt(self):
            return "PROMPT"

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            mapping = {
                10: "PROMPT_A",
                11: "PROMPT_B",
                90: "Feedback: too generic. ",
                91: "[RESULT] 2",
            }
            return "".join(mapping[token] for token in token_ids)

    class FakeModel:
        device = "cpu"
        config = object()

        def generate(self, input_ids, images, **kwargs):
            assert input_ids.tolist() == [[10, 11]]
            return FakeTensor([[10, 11, 90, 91]])

    monkeypatch.setattr(PrometheusVisionScorer, "_ensure_loaded", lambda self: None)
    scorer._Image = FakeImageModule
    scorer._process_images = lambda images, processor, config: FakeTensor([[1, 2]])
    scorer._image_processor = object()
    scorer._model = FakeModel()
    scorer._torch = FakeTorch()
    scorer._conv_templates = {"vicuna_v1": FakeConv()}
    scorer._DEFAULT_IMAGE_TOKEN = "<image>"
    scorer._tokenizer_image_token = (
        lambda full_prompt, tokenizer, image_token_index, return_tensors=None: FakeTensor([10, 11])
    )
    scorer._tokenizer = FakeTokenizer()
    scorer._IMAGE_TOKEN_INDEX = 0

    raw_text = scorer.generate(
        image_path="/tmp/slide.png",
        prompt="Judge this answer.",
    )

    assert raw_text == "Feedback: too generic. [RESULT] 2"
