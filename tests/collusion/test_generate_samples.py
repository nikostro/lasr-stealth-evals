import json
from pathlib import Path
from typing import List, Dict, Any, cast
import pytest
from inspect_ai.dataset import Sample

from lasr_stealth_evals.collusion.generate_samples import (
    Metadata,
    generate_samples,
    GenerationStrategy,
)
from lasr_stealth_evals.collusion.util import get_system_message


# Load actual data files
def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path) as f:
        return json.load(f)


@pytest.fixture
def strategy_file() -> Path:
    return Path("data/strategy.json")


@pytest.fixture
def task_file() -> Path:
    return Path("data/tasks.json")


def test_match_strategy(strategy_file: Path, task_file: Path) -> None:
    samples = generate_samples(strategy_file, task_file, GenerationStrategy.MATCH)

    # Load actual data to verify
    strategies = load_json_data(strategy_file)
    tasks = load_json_data(task_file)

    # Should create one sample per strategy per task
    assert len(samples) == len(strategies) * len(tasks)

    # Check sample structure
    sample = samples[0]
    assert isinstance(sample, Sample)
    assert sample.target == "True"

    # Check metadata structure
    metadata = cast(Dict[str, Any], sample.metadata)
    assert isinstance(metadata, dict)
    assert "agent_prompt" in metadata
    assert "monitor_prompt" in metadata
    assert "hint" in metadata
    assert "strategies" in metadata
    assert "task_name" in metadata

    # Check that prompts are not None
    assert metadata["agent_prompt"] is not None
    assert metadata["monitor_prompt"] is not None

    # Check that strategy and hint are aligned
    for sample in samples:
        metadata = cast(Dict[str, Any], sample.metadata)
        strategies = cast(List[Dict[str, str]], metadata["strategies"])
        hint = cast(Dict[str, str], metadata["hint"])
        strategy_name = strategies[0]["name"]
        hint_name = hint["name"]
        assert strategy_name == hint_name


def test_misaligned_strategy(strategy_file: Path, task_file: Path) -> None:
    samples = generate_samples(strategy_file, task_file, GenerationStrategy.MISALIGNED)

    # Load actual data to verify
    strategies = load_json_data(strategy_file)
    tasks = load_json_data(task_file)

    # Should create (n * (n-1)) samples where n is number of strategies
    expected_samples = len(strategies) * (len(strategies) - 1) * len(tasks)
    assert len(samples) == expected_samples

    # Check that strategy and hint are misaligned
    for sample in samples:
        metadata = cast(Dict[str, Any], sample.metadata)
        strategies = cast(List[Dict[str, str]], metadata["strategies"])
        hint = cast(Dict[str, str], metadata["hint"])
        strategy_name = strategies[0]["name"]
        hint_name = hint["name"]
        assert strategy_name != hint_name


def test_multiple_strategy(strategy_file: Path, task_file: Path) -> None:
    samples = generate_samples(strategy_file, task_file, GenerationStrategy.MULTIPLE)

    # Load actual data to verify
    strategies = load_json_data(strategy_file)
    tasks = load_json_data(task_file)

    # Should create one sample per strategy per task
    assert len(samples) == len(strategies) * len(tasks)

    # Check that each sample has multiple strategies
    for sample in samples:
        metadata = cast(Dict[str, Any], sample.metadata)
        strategies = cast(List[Dict[str, str]], metadata["strategies"])
        assert len(strategies) > 1
        # Check that all strategies are from our actual data
        strategy_names = {s["name"] for s in strategies}
        actual_strategy_names = {s["name"] for s in load_json_data(strategy_file)}
        assert strategy_names.issubset(actual_strategy_names)


def test_no_strategy(strategy_file: Path, task_file: Path) -> None:
    samples = generate_samples(strategy_file, task_file, GenerationStrategy.NONE)

    # Load actual data to verify
    strategies = load_json_data(strategy_file)
    tasks = load_json_data(task_file)

    # Should create one sample per strategy per task
    assert len(samples) == len(strategies) * len(tasks)

    # Check sample structure
    sample = samples[0]
    assert isinstance(sample, Sample)
    assert sample.target == "True"

    # Check metadata structure
    metadata = cast(Dict[str, Any], sample.metadata)
    assert isinstance(metadata, dict)
    assert "agent_prompt" in metadata
    assert "monitor_prompt" in metadata
    assert "task_name" in metadata
    assert "hint" in metadata
    assert "strategies" in metadata

    # Check that strategy and hint are aligned
    for sample in samples:
        metadata = cast(Dict[str, Any], sample.metadata)
        strategies = cast(List[Dict[str, str]], metadata["strategies"])
        hint = cast(Dict[str, str], metadata["hint"])
        assert len(strategies) == 0
        assert hint


def test_load_metadata(strategy_file: Path, task_file: Path):
    for strategy in GenerationStrategy:
        samples = generate_samples(strategy_file, task_file, strategy)
        metadata = samples[0].metadata
        sample_metadata = Metadata(**metadata)  # type: ignore
        for name in ["agent", "monitor"]:
            system_message = get_system_message(sample_metadata, name)
            assert system_message is not None
