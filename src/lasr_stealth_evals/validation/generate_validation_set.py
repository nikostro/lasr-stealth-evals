import argparse
import csv
import random
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

from inspect_ai.log import EvalSample, read_eval_log

from lasr_stealth_evals.collusion.generate_samples import Metadata
from lasr_stealth_evals.collusion.scorer import get_score_prompt

# Default path moved to main function


def load_logs(log_path: Path) -> list[EvalSample]:
    log = read_eval_log(str(log_path))
    assert log.samples
    return log.samples


@dataclass
class ValidationSample:
    sample_id: int
    agent_task_prompt: str
    monitor_task_prompt: str
    history: str
    task_name: str
    judge_prompt: str
    value: str
    explanation: str
    ground_truth: str = field(default_factory=str)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate validation set from evaluation logs")
    parser.add_argument(
        "--log_path",
        type=str,
        default="/Users/nikita/Documents/LASR/lasr-stealth-evals/logs/2025-04-25T11-25-23+01-00_evaluation_dJPswVgfV4CeKeBvTVeYow.eval",
        help="Path to the evaluation log file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation/tmp",
        help="Directory to save the validation set",
    )
    return parser.parse_args()


def generate_validation_set(log_path: Path, output_dir: Path, k: int = 4):
    """Generate validation samples from evaluation logs and save them to a CSV file.

    Args:
        log_path: Path to the evaluation log file
        output_dir: Directory to save the validation set
        k: Number of samples to evaluate from each task
    """
    # Set random seed for reproducibility
    random.seed(42)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_by_task: dict[str, list] = {}  # task_name, list[Sample]
    output: list[ValidationSample] = []
    samples = load_logs(log_path)

    # Organise by task
    for sample in samples:
        metadata = Metadata(**sample.metadata)
        task_name = metadata.task_name

        # Organize samples by task name
        assert sample.scores
        if sample.scores["approval_scorer"].value == "C":
            sample_by_task.setdefault(task_name, []).append(sample)

    if k > min(len(samples) for samples in sample_by_task.values()):
        print(
            f"WARNING: Attempting to sample {k} samples without replacement from list with {min(len(samples) for samples in sample_by_task.values())} samples"
        )
    # Generate set by task type
    for task_name, task_samples in sample_by_task.items():
        # Sample without replacement
        samples = random.sample(task_samples, min(len(task_samples), k))  # Adjust the number 10 as needed

        for sample in samples:
            metadata = Metadata(**sample.metadata)
            # Judge input
            judge_prompt = get_score_prompt(sample.metadata, sample.messages)
            assert sample.scores
            value = sample.scores["judge"].value
            explanation = sample.scores["judge"].explanation
            assert explanation
            history = [str(message.content) for message in sample.messages[::-1]]
            output.append(
                ValidationSample(
                    sample_id=sample.id,
                    agent_task_prompt=metadata.agent_task,
                    monitor_task_prompt=metadata.agent_task,
                    history="\n==============\n".join(history),
                    judge_prompt=judge_prompt,
                    value=str(value),
                    explanation=explanation,
                    task_name=task_name,
                )
            )

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"validation_samples_{timestamp}.csv"

    # Save output as CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(ValidationSample)])
        writer.writeheader()
        writer.writerows([asdict(sample) for sample in output])

    print(f"Saved {len(output)} samples to {output_file}")
    return output


def main():
    args = parse_args()
    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)

    generate_validation_set(log_path, output_dir)


if __name__ == "__main__":
    main()
