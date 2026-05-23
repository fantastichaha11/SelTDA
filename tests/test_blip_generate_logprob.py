import pytest
import torch

from models.blip import blip_decoder


@pytest.mark.slow
def test_generate_returns_logprob_finite_and_negative():
    model = blip_decoder(
        pretrained="",
        image_size=384,
        vit="base",
        med_config="configs/med_config.json",
        prompt="Question:",
    )
    model.eval()
    images = torch.randn(2, 3, 384, 384)
    captions, logprobs = model.generate(
        images,
        sample=True,
        top_p=0.9,
        max_length=20,
        min_length=5,
        return_logprob=True,
    )
    assert len(captions) == 2 and len(logprobs) == 2
    assert all(lp <= 0.0 and lp > -50.0 for lp in logprobs)


@pytest.mark.slow
def test_generate_default_signature_unchanged():
    model = blip_decoder(
        pretrained="",
        image_size=384,
        vit="base",
        med_config="configs/med_config.json",
        prompt="Question:",
    )
    model.eval()
    images = torch.randn(1, 3, 384, 384)
    captions = model.generate(
        images,
        sample=True,
        top_p=0.9,
        max_length=20,
        min_length=5,
    )
    assert isinstance(captions, list) and isinstance(captions[0], str)
