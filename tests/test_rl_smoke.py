import torch

from orchestration.rl_teacher_loop import kl_beta_for_round, should_continue
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


def test_should_continue_above_delta():
    assert should_continue(acc_curr=62.0, acc_prev=61.0, delta=0.5) is True


def test_should_stop_below_delta():
    assert should_continue(acc_curr=61.2, acc_prev=61.0, delta=0.5) is False


def test_kl_beta_decays_per_round():
    assert abs(kl_beta_for_round(beta0=0.1, gamma=0.7, r=1) - 0.1) < 1e-9
    assert abs(kl_beta_for_round(beta0=0.1, gamma=0.7, r=2) - 0.07) < 1e-9
