import csv
import tempfile
from pathlib import Path

import pytest

from lasr_stealth_evals.validation.generate_validation_set import ValidationSample, generate_validation_set


@pytest.fixture
def sample_log_path() -> Path:
    """Path to the actual log file"""
    return Path(
        "/Users/nikita/Documents/LASR/lasr-stealth-evals/logs/2025-04-28T12-21-34+01-00_evaluation_fa7JwjRehyKvzLnVv7Cqm4.eval"
    )


def test_integration_with_real_log(sample_log_path: Path) -> None:
    """Integration test using the actual log file if it exists"""
    if not sample_log_path.exists():
        pytest.skip(f"Log file {sample_log_path} does not exist")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Run the function with the real log file
        result = generate_validation_set(sample_log_path, output_dir)

        # Basic validation of results
        assert len(result) > 0
        for sample in result:
            assert isinstance(sample, ValidationSample)
            assert sample.judge_prompt
            assert sample.value
            assert sample.explanation

        # Verify there is exactly one file in the output directory
        files_in_dir = list(output_dir.iterdir())
        assert len(files_in_dir) == 1, f"Expected exactly one file in output directory, found {len(files_in_dir)}"
        output_file = files_in_dir[0]

        # Verify CSV can be loaded and has the expected number of rows
        with open(output_file) as f:
            reader = csv.DictReader(f)
            data = list(reader)
            assert len(data) == len(result)


if __name__ == "__main__":
    pytest.main(["-v"])
