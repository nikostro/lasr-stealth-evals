# %%
import csv
import json
import random
from typing import Tuple

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalSample
from inspect_ai.scorer import Metric, SampleScore, ValueToFloat, accuracy, match, metric, stderr, value_to_float
from inspect_ai.solver import generate

from lasr_stealth_evals.collusion.scorer import get_score_prompt
from lasr_stealth_evals.validation.generate_validation_set import ValidationSample

sample_path = "/Users/nikita/Documents/LASR/lasr-stealth-evals/src/lasr_stealth_evals/validation/samples.json"
validation_set_path = "/Users/nikita/Downloads/Validation Dataset - validation_samples_2025-04-29_18-17-26.csv"


def get_dataset(
    sample_path: str, validation_set_path: str, n_test: int = 15, seed: int = 42
) -> Tuple[list[Sample], list[Sample]]:
    random.seed(seed)

    with open(sample_path, "r") as f:
        input_samples = {int(sample.id): sample for sample in [EvalSample(**sample) for sample in json.load(f)]}

    with open(validation_set_path, "r") as f:
        validation_samples: dict[int, ValidationSample] = {}
        for row in csv.DictReader(f):
            row["sample_id"] = int(row["sample_id"])
            validation_samples[row["sample_id"]] = ValidationSample(**row)  # type: ignore

    # Ensure we have matching samples
    assert set(input_samples.keys()) == set(
        validation_samples.keys()
    ), "Sample IDs don't match between input and validation sets"

    # Get all valid sample IDs (those with y/n ground truth)
    valid_sample_ids = [
        sample_id for sample_id in input_samples if validation_samples[sample_id].ground_truth in ["y", "n"]
    ]

    # Randomly select test samples if n_test > 0
    test_sample_ids = set(random.sample(valid_sample_ids, min(n_test, len(valid_sample_ids)))) if n_test > 0 else set()
    train_sample_ids = set(valid_sample_ids) - test_sample_ids

    def gt_to_grade(gt: str) -> str:
        return "GRADE: C" if gt == "y" else "GRADE: I"

    def create_samples(sample_ids: set[int]) -> list[Sample]:
        return [
            Sample(
                id=sample_id,
                input=get_score_prompt(input_samples[sample_id].metadata, input_samples[sample_id].messages),
                target=gt_to_grade(validation_samples[sample_id].ground_truth),
            )
            for sample_id in sample_ids
        ]

    train_samples = create_samples(train_sample_ids)
    test_samples = create_samples(test_sample_ids)

    return train_samples, test_samples


# %%
@metric
def precision(to_float: ValueToFloat = value_to_float()) -> Metric:
    r"""Compute proportion of total answers which are correct.

    Args:
       to_float: Function for mapping `Value` to float for computing
          metrics. The default `value_to_float()` maps CORRECT ("C") to 1.0,
          INCORRECT ("I") to 0, PARTIAL ("P") to 0.5, and NOANSWER ("N") to 0,
          casts numeric values to float directly, and prints a warning and returns
          0 if the Value is a complex object (list or dict).

    Returns:
       Accuracy metric
    """

    def metric(scores: list[SampleScore]) -> float:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for item in scores:
            assert item.score.answer is not None
            if "GRADE: C" in item.score.answer:
                if item.score.value == "C":
                    tp += 1
                else:
                    fp += 1
            elif "GRADE: I" in item.score.answer:
                if item.score.value == "I":
                    tn += 1
                else:
                    fn += 1
            else:
                print(f"Grade C and I not found in: {item.score.answer}")
        return tp / (tp + fp)

    return metric


@task
def validate_model():
    train_samples, test_samples = get_dataset(sample_path, validation_set_path)
    return Task(
        dataset=train_samples,
        model="openrouter/openai/gpt-4.1-mini",
        solver=[generate()],
        scorer=match(),
        metrics=[
            accuracy(),
            stderr(),
            precision(),
        ],
    )
