import torch

from train_vqg_rl import grpo_loss, group_advantages


def test_group_advantages_normalizes_within_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = group_advantages(rewards)
    assert abs(float(adv.mean())) < 1e-6
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-5


def test_grpo_loss_sign():
    logp = torch.tensor([-1.0, -2.0])
    ref_logp = torch.tensor([-1.0, -1.0])
    adv = torch.tensor([1.0, -1.0])
    loss = grpo_loss(logp, ref_logp, adv, kl_beta=0.1)
    assert torch.isfinite(loss)
