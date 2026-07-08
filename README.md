[![Conference](https://img.shields.io/badge/CVPR-2023-blue)](https://openaccess.thecvf.com/content/CVPR2023/html/Khan_Q_How_To_Specialize_Large_Vision-Language_Models_to_Data-Scarce_VQA_CVPR_2023_paper.html)
[![Paper](http://img.shields.io/badge/paper-arxiv.2306.03932-B31B1B.svg)](https://arxiv.org/abs/2306.03932)

# SelTDA
This repository will hold the official code of SelTDA, the self-training framework introduced in our CVPR 2023 paper "Q: How to Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!".


![seltda_teaser](https://user-images.githubusercontent.com/4918041/225918833-7d744775-260a-4bc3-a642-7279531b5b07.png)

## Environment
```bash
conda env create -f environment.yaml
```

## Data
### Downloads and Preprocessing
- [PathVQA](https://github.com/UCSD-AI4H/PathVQA)
    - then use `convert_pathvqa.py`
- [RSVQA](https://rsvqa.sylvainlobry.com/)
    - then use `convert_rsvqa.py`
- OK-VQA and A-OKVQA (use [LAVIS](https://github.com/salesforce/LAVIS))
    - LAVIS should automatically put them in the correct format, but if not, you can use `convert_okvqa.py`
- [VQA Counterexamples](https://github.com/cdancette/detect-shortcuts)
    - then use `convert_vqa_ce.py`
- [AdVQA](https://adversarialvqa.org/download.html)
    - then use `convert_advqa.py`
- [VQA Rephrasings](https://facebookresearch.github.io/VQA-Rephrasings/)
    - then use `convert_vqa_rephrasings.py`

In general, the code expects that each VQA dataset is represented by a single JSON object that is a list of dictionaries. In `schemas.py`, we provide Pydantic models which you can use to define your own datasets or verify that the data is in the correct format. 

## Experiments
See the `examples/` directory to see examples of:
- training the teacher 
    - `examples/train_teacher.sh`
- generating synthetic data with the teacher
    - `examples/generate_synthetic_data.sh`
- self-training with the synthetic data
    - `examples/self_train_synthetic.sh`
- evaluations
    - `examples/evaluate.sh`

## Citation
```
@InProceedings{Khan_2023_CVPR,
    author    = {Khan, Zaid and BG, Vijay Kumar and Schulter, Samuel and Yu, Xiang and Fu, Yun and Chandraker, Manmohan},
    title     = {Q: How To Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15005-15015}
}
```


## Acknowledgements
This code is heavily based on [salesforce/BLIP](https://github.com/salesforce/BLIP).

```bash
cd ~/SelTDA
conda activate vqa
export PYTHONNOUSERSITE=1

# 1) Data + checkpoint
bash dataset.sh

# 2) Convert A-OKVQA
python convert_aokvqa.py \
  --config configs/aokvqa.yaml \
  --overrides "vqa_root='$(pwd)/datasets/coco2017'" "ann_root='$(pwd)/datasets/aokvqa'"

# 3) Generate pseudo-QA từ unlabeled COCO
python generate_questions.py \
  --config configs/generate_questions_coco.yaml \
  --overrides \
    "image_folder='$(pwd)/datasets/coco2017/unlabeled2017'" \
    "output_folder='$(pwd)/datasets/aokvqa'" \
    "pretrained='$(pwd)/cache/teacher_weights/checkpoint_04.pth'" \
    "output_annotations_name=synthetic_data_raw.json" \
    "multimodal_encoder_decoder_config='$(pwd)/configs/med_config.json'" \
    "questions_per_image=2" \
    "max_length=40" \
    "batch_size=16"

# 4) Filter
python filter_pseudo.py \
  --config configs/filter_pseudo.yaml \
  --overrides \
    "input='$(pwd)/datasets/aokvqa/synthetic_data_raw.json'" \
    "image_root='$(pwd)/datasets/coco2017'" \
    "output='$(pwd)/datasets/aokvqa/synthetic_data.json'" \
    "report='$(pwd)/datasets/aokvqa/filter_report.json'"

# 5) Train student
python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
  --output_dir=cache/self_trained_weights \
  --config configs/aokvqa.yaml \
  --overrides \
    "vqa_root='$(pwd)/datasets/coco2017'" \
    "ann_root='$(pwd)/datasets/aokvqa'" \
    "train_files=[train,synthetic_data]" \
    "truncate_train_dataset_to=34000" \
    "wandb=false"

# Resume after interruption (latest checkpoint in output_dir):
# python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
#   --output_dir=cache/self_trained_weights \
#   --config configs/aokvqa.yaml \
#   --resume auto \
#   --overrides \
#     "vqa_root='$(pwd)/datasets/coco2017'" \
#     "ann_root='$(pwd)/datasets/aokvqa'" \
#     "train_files=[train,synthetic_data]" \
#     "truncate_train_dataset_to=34000" \
#     "wandb=false"
```

## Prometheus-Vision Judge For PathVQA

The Prometheus judge path is separate from pseudo-label filtering. It uses PathVQA train annotations to build positive/negative judge adaptation data, evaluates pretrained or adapted judges on PathVQA val, and can provide a frozen reward signal for GRPO-style VQG teacher training.

Mock eval:

```bash
python scripts/eval_prometheus_judge.py \
  --config configs/prometheus_judge_pathvqa.yaml \
  --max-examples 4 \
  --mock-score 3
```

Judge adaptation export:

```bash
python scripts/train_prometheus_judge.py \
  --config configs/prometheus_judge_pathvqa.yaml \
  --backend dry_run
```

Mock GRPO reward loop:

```bash
python scripts/train_teacher_grpo.py \
  --config configs/grpo_teacher_pathvqa_prometheus.yaml \
  --mock-judge-score 4
```

For real Prometheus-Vision scoring, install the Prometheus/LLaVA runtime and set `PROMETHEUS_VISION_MODEL` to the pretrained or adapted checkpoint. Keep `image_pool.use_ground_truth_qa: false` for teacher reward optimization so PathVQA train annotations are used only to enumerate images.
