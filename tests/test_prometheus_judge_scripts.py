from omegaconf import OmegaConf


def test_prometheus_judge_config_loads():
    cfg = OmegaConf.load("configs/prometheus_judge_pathvqa.yaml")

    assert cfg.data.train_annotations == "datasets/pathvqa/train.json"
    assert cfg.data.val_annotations == "datasets/pathvqa/val.json"
    assert cfg.prompt.no_reference is True
    assert cfg.eval.split == "val"
    assert cfg.train.use_pseudo_qa is False


def test_grpo_teacher_config_uses_pathvqa_train_images_only():
    cfg = OmegaConf.load("configs/grpo_teacher_pathvqa_prometheus.yaml")

    assert cfg.image_pool.name == "pathvqa_train"
    assert cfg.image_pool.annotations == "datasets/pathvqa/train.json"
    assert cfg.image_pool.use_ground_truth_qa is False
    assert cfg.reward.judge_config == "configs/prometheus_judge_pathvqa.yaml"
