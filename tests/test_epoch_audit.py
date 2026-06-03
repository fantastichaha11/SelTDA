"""Epoch audit wiring tests (mocked judges)."""

from unittest.mock import MagicMock

import torch

from filtering.audit import (
    EpochAuditResult,
    detect_hacking,
    judge_score,
    run_epoch_audit,
    term_gain_shares,
)


def test_term_gain_shares_itm_dominant():
    prev = {"itm": 0.2, "grounding": 0.5}
    curr = {"itm": 0.8, "grounding": 0.52}
    weights = {"itm": 1.0, "grounding": 1.0}
    shares = term_gain_shares(prev, curr, weights)
    assert shares["itm"] > shares["grounding"]


def test_run_epoch_audit_mocked(monkeypatch):
    class FakeDS:
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.zeros(3, 8, 8), f"img{idx}.jpg"

    itm = MagicMock()
    itm.match_prob.return_value = 0.9
    vqa = MagicMock()
    vqa.answer_question.return_value = "blue"

    config = MagicMock()
    config.teacher.med_config = "configs/med_config.json"
    config.teacher.image_size = 384
    config.audit.itm_vit = "large"

    def gen_fn(policy, image_tensor, device):
        return "What color is it? blue"

    def terms_fn(path, q, a):
        return 1.5, {"itm": 0.8, "grounding": 0.5}

    monkeypatch.setattr(
        "filtering.audit.build_audit_judges",
        lambda *a, **k: (itm, vqa),
    )

    result = run_epoch_audit(
        MagicMock(),
        FakeDS(),
        device="cpu",
        n_audit=2,
        seed=0,
        generate_fn=gen_fn,
        reward_terms_fn=terms_fn,
        config=config,
    )
    assert isinstance(result, EpochAuditResult)
    assert result.n_samples == 2
    assert abs(result.mean_itm_judge - 0.9) < 1e-6
    assert result.mean_reward == 1.5


def test_detect_hacking_reward_up_judge_down():
    assert detect_hacking(0.1, -0.05, 0.02, {"itm": 0.1}, 0.6)


def test_resolve_backfill_ckpt_epoch0_fallback(tmp_path):
    from omegaconf import OmegaConf

    from train_vqg_rl import _resolve_backfill_ckpt

    fb = tmp_path / "ckpt04.pth"
    fb.write_bytes(b"x")
    config = OmegaConf.create(
        {
            "audit": {"backfill_epoch0_ckpt": str(fb)},
            "teacher": {"pretrained": str(fb)},
        }
    )
    got = _resolve_backfill_ckpt(0, 1, tmp_path, config)
    assert got == fb


def test_resolve_backfill_ckpt_prefers_epoch_file(tmp_path):
    from omegaconf import OmegaConf

    from train_vqg_rl import _resolve_backfill_ckpt

    ep1 = tmp_path / "teacher_1_epoch1.pth"
    ep1.write_bytes(b"x")
    config = OmegaConf.create(
        {
            "audit": {"backfill_epoch0_ckpt": "missing"},
            "teacher": {"pretrained": "missing"},
        }
    )
    assert _resolve_backfill_ckpt(1, 1, tmp_path, config) == ep1
