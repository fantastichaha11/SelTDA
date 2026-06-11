from filtering.scorers_vqascore import score_vqascore


class StubVQAScorer:
    def __init__(self, score: float = 0.9):
        self.score = score
        self.last_paths = None
        self.last_texts = None

    def score_pairs(self, image_paths, texts):
        self.last_paths = list(image_paths)
        self.last_texts = list(texts)
        return [self.score] * len(image_paths)


def test_score_vqascore_formats_qa_text():
    scorer = StubVQAScorer(score=0.85)
    record = {"question": "what is this?", "answer": ["dog"], "image": "vg/img.jpg"}
    s = score_vqascore(record, "/tmp/vg/img.jpg", scorer)
    assert s == 0.85
    assert scorer.last_paths == ["/tmp/vg/img.jpg"]
    assert scorer.last_texts == ["what is this? dog"]


def test_score_vqascore_handles_string_answer():
    scorer = StubVQAScorer(score=0.7)
    record = {"question": "is it raining?", "answer": "yes", "image": "x.jpg"}
    s = score_vqascore(record, "images/x.jpg", scorer)
    assert s == 0.7
    assert scorer.last_texts == ["is it raining? yes"]
