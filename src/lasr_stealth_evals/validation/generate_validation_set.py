import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from inspect_ai.log import EvalSample, read_eval_log

from lasr_stealth_evals.collusion.scorer import get_score_prompt

# Default path moved to main function


def load_logs(log_path: Path) -> list[EvalSample]:
    log = read_eval_log(str(log_path))
    assert log.samples
    return log.samples


@dataclass
class ValidationSample:
    prompt: str
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


def generate_validation_set(log_path: Path, output_dir: Path):
    """Generate validation samples from evaluation logs and save them to a JSON file.

    Args:
        log_path: Path to the evaluation log file
        output_dir: Directory to save the validation set
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    output: list[ValidationSample] = []
    samples = load_logs(log_path)
    for sample in samples:
        # Judge input
        score_prompt = get_score_prompt(sample.metadata, sample.messages)

        # Judge decisions
        assert sample.scores
        value = sample.scores["judge"].value
        explanation = sample.scores["judge"].explanation
        assert explanation
        output.append(ValidationSample(prompt=score_prompt, value=str(value), explanation=explanation))

    # Save output as JSON
    output_file = output_dir / "validation_samples.json"
    with open(output_file, "w") as f:
        json.dump([asdict(sample) for sample in output], f, indent=2)

    print(f"Validation samples saved to {output_file}")
    return output


def main():
    args = parse_args()
    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)

    generate_validation_set(log_path, output_dir)


if __name__ == "__main__":
    main()
