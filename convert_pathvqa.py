import json
from pathlib import Path
from dataclasses import dataclass, field
from argparse import ArgumentParser
import schemas
from pydantic import BaseModel, validator
from tqdm import tqdm
import logging
from typing import Tuple, List, Union, Dict

PATHVQA_ROOT = Path("/net/acadia4a/data/zkhan/pathvqa")


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s -  %(name)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_json(_path: Path):
    with open(_path, "r") as f:
        return json.load(f)


def write_json(_path: Path, data: dict):
    with open(_path, "w") as f:
        json.dump(data, f)


# For evaluting this code, use an exact match evaluation.
# There's only one answer per question in PathVQA, so you
# can't use the VQAv2 evaluation code.
@dataclass
class OutputAnnotations:
    train: Path = field(default_factory=lambda: PATHVQA_ROOT / "train.json")
    val: Path = field(default_factory=lambda: PATHVQA_ROOT / "val.json")
    test: Path = field(default_factory=lambda: PATHVQA_ROOT / "test.json")
    answer_list: Path = field(default_factory=lambda: PATHVQA_ROOT / "answer_list.json")
    # This can be easily used to generate questions on images unused during training.
    test_val_combined: Path = field(
        default_factory=lambda: PATHVQA_ROOT / "test_val_combined.json"
    )


@dataclass
class Config:
    raw_annotations: Path = field(default_factory=lambda: PATHVQA_ROOT / "all_data.json")
    output_annotations: OutputAnnotations = field(default_factory=OutputAnnotations)
    pathvqa_images_dir: Path = field(default_factory=lambda: PATHVQA_ROOT / "images")


def make_config(pathvqa_root: Path, images_dir: Path = None) -> Config:
    pathvqa_root = pathvqa_root.resolve()
    images_dir = (images_dir or pathvqa_root / "images").resolve()
    return Config(
        raw_annotations=pathvqa_root / "all_data.json",
        output_annotations=OutputAnnotations(
            train=pathvqa_root / "train.json",
            val=pathvqa_root / "val.json",
            test=pathvqa_root / "test.json",
            answer_list=pathvqa_root / "answer_list.json",
            test_val_combined=pathvqa_root / "test_val_combined.json",
        ),
        pathvqa_images_dir=images_dir,
    )


def parse_args():
    parser = ArgumentParser(description="Convert PathVQA into SelTDA JSON files.")
    parser.add_argument(
        "--pathvqa-root",
        type=Path,
        default=PATHVQA_ROOT,
        help="Directory containing all_data.json and where converted JSON files are written.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing PathVQA images/. Defaults to <pathvqa-root>/images.",
    )
    return parser.parse_args()


class PathVQARecord_SuffixQA(BaseModel):
    image: str
    question: str
    answer: str


class PathVQARecord_SuffixVQA(BaseModel):
    answer_type: str
    img_id: str
    label: Dict[str, int]
    question_id: int
    question_type: str
    sent: str


class PathVQADump(BaseModel):
    test_qa: List[PathVQARecord_SuffixQA]
    test_vqa: List[PathVQARecord_SuffixVQA]
    train_qa: List[PathVQARecord_SuffixQA]
    train_vqa: List[PathVQARecord_SuffixVQA]
    val_qa: List[PathVQARecord_SuffixQA]
    val_vqa: List[PathVQARecord_SuffixVQA]


def convert_pathvqa_record_to_train_record(
    record_suffixqa: PathVQARecord_SuffixQA, record_suffixvqa: PathVQARecord_SuffixVQA
) -> schemas.TrainingRecord:
    return schemas.TrainingRecord(
        question=record_suffixqa.question,
        answer=[record_suffixqa.answer],
        image=record_suffixqa.image,
        dataset="pathvqa",
        question_id=record_suffixvqa.question_id,
    )


def convert_pathvqa_record_to_evaluation_record(
    record_suffixqa: PathVQARecord_SuffixQA, record_suffixvqa: PathVQARecord_SuffixVQA
) -> schemas.MinimalEvaluationRecord:
    return schemas.MinimalEvaluationRecord(
        question=record_suffixqa.question,
        answer=record_suffixqa.answer,
        image=record_suffixqa.image,
        dataset="pathvqa",
        question_id=record_suffixvqa.question_id,
        question_type=record_suffixvqa.question_type,
        answer_type=record_suffixvqa.answer_type,
    )


def make_training_records(pathvqa_dump: PathVQADump) -> List[schemas.TrainingRecord]:
    training_records = []
    for record_suffixqa, record_suffixvqa in zip(
        pathvqa_dump.train_qa, pathvqa_dump.train_vqa
    ):
        training_records.append(
            convert_pathvqa_record_to_train_record(record_suffixqa, record_suffixvqa)
        )
    return training_records


def make_validation_records(
    pathvqa_dump: PathVQADump,
) -> List[schemas.MinimalEvaluationRecord]:
    validation_records = []
    for record_suffixqa, record_suffixvqa in zip(
        pathvqa_dump.val_qa, pathvqa_dump.val_vqa
    ):
        validation_records.append(
            convert_pathvqa_record_to_evaluation_record(
                record_suffixqa, record_suffixvqa
            )
        )
    return validation_records


def make_testing_records(
    pathvqa_dump: PathVQADump,
) -> List[schemas.MinimalEvaluationRecord]:
    testing_records = []
    for record_suffixqa, record_suffixvqa in zip(
        pathvqa_dump.test_qa, pathvqa_dump.test_vqa
    ):
        testing_records.append(
            convert_pathvqa_record_to_evaluation_record(
                record_suffixqa, record_suffixvqa
            )
        )
    return testing_records


def redirect_image_and_verify(
    image_dir: Path,
    record: Union[schemas.TrainingRecord, schemas.MinimalEvaluationRecord],
) -> None:
    image_name = record.image
    split, *_ = image_name.split("_")

    path = f"{split}/{image_name}.jpg"

    record.image = path
    try:
        assert (image_dir / record.image).exists(), f"Image {path} does not exist"
    except AssertionError:
        import ipdb

        ipdb.set_trace()


def make_answer_list(test_records: List[schemas.MinimalEvaluationRecord]) -> List[str]:
    answer_list = []
    for record in test_records:
        answer_list.append(record.answer)
    return list(set(answer_list))


def filter_to_unique_images(
    records: List[schemas.TrainingRecord],
) -> List[schemas.TrainingRecord]:
    unique_images = set()
    filtered_records = []
    for record in records:
        if record.image not in unique_images:
            unique_images.add(record.image)
            filtered_records.append(record)
    return filtered_records


if __name__ == "__main__":
    args = parse_args()
    conf = make_config(args.pathvqa_root, args.images_dir)
    pathvqa_dump = PathVQADump.parse_obj(load_json(conf.raw_annotations))
    training_records = make_training_records(pathvqa_dump)
    logger.info("Made %d training records", len(training_records))
    validation_records = make_validation_records(pathvqa_dump)
    logger.info("Made %d validation records", len(validation_records))
    testing_records = make_testing_records(pathvqa_dump)
    logger.info("Made %d testing records", len(testing_records))
    logger.info("Verifying all records")
    for record in tqdm(training_records + validation_records + testing_records):
        redirect_image_and_verify(conf.pathvqa_images_dir, record)

    answer_list = make_answer_list(testing_records)
    logger.info("Made answer list with %d answers", len(answer_list))

    write_json(
        conf.output_annotations.train, [record.dict() for record in training_records]
    )
    write_json(
        conf.output_annotations.val, [record.dict() for record in validation_records]
    )
    write_json(
        conf.output_annotations.test, [record.dict() for record in testing_records]
    )
    write_json(conf.output_annotations.answer_list, answer_list)
    write_json(
        conf.output_annotations.test_val_combined,
        [
            record.dict()
            for record in filter_to_unique_images(testing_records + validation_records)
        ],
    )
