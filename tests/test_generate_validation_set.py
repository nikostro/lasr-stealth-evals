import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lasr_stealth_evals.validation.generate_validation_set import ValidationSample, generate_validation_set


@pytest.fixture
def mock_eval_sample():
    """Create a mock EvalSample for testing"""
    mock_sample = MagicMock()
    mock_sample.metadata = {"task": "test task", "model": "test model"}
    mock_sample.messages = [{"role": "user", "content": "test message"}]
    mock_sample.scores = {"judge": MagicMock(value="Pass", explanation="This is a test explanation")}
    return mock_sample


@pytest.fixture
def sample_log_path() -> Path:
    """Path to the actual log file"""
    return Path(
        "/Users/nikita/Documents/LASR/lasr-stealth-evals/logs/2025-04-25T11-25-23+01-00_evaluation_dJPswVgfV4CeKeBvTVeYow.eval"
    )


@patch("lasr_stealth_evals.validation.generate_validation_set.load_logs")
@patch("lasr_stealth_evals.validation.generate_validation_set.get_score_prompt")
def test_generate_validation_set(mock_get_score_prompt: Any, mock_load_logs: Any, mock_eval_sample: Any) -> None:
    """Test the generate_validation_set function with mocked data"""
    # Setup
    mock_load_logs.return_value = [mock_eval_sample]
    mock_get_score_prompt.return_value = "test prompt"

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Run the function
        result = generate_validation_set(Path("mock_log_path"), output_dir)

        # Verify results
        assert len(result) == 1
        assert isinstance(result[0], ValidationSample)
        assert result[0].prompt == "test prompt"
        assert result[0].value == "Pass"
        assert result[0].explanation == "This is a test explanation"

        # Check if the JSON file was created
        output_file = output_dir / "validation_samples.json"
        assert output_file.exists()

        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["prompt"] == "test prompt"
            assert data[0]["value"] == "Pass"
            assert data[0]["explanation"] == "This is a test explanation"


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
            assert sample.prompt
            assert sample.value
            assert sample.explanation

        # Check if the JSON file was created
        output_file = output_dir / "validation_samples.json"
        assert output_file.exists()

        # Verify JSON can be loaded
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == len(result)


if __name__ == "__main__":
    pytest.main(["-v"])
