import json
from pathlib import Path
from typing import List, Dict, Any, cast
import pytest
from inspect_ai.dataset import Sample

from lasr_stealth_evals.collusion.generate_samples import (
    generate_samples,
    GenerationStrategy,
)

# Test data
TEST_STRATEGIES: List[Dict[str, str]] = [
    {"name": "pathos", "description": "Use emotional appeals", "hint": "You care about emotions"},
    {"name": "logos", "description": "Use logical appeals", "hint": "You value logic"},
]

TEST_TASKS: List[Dict[str, str]] = [
    {
        "name": "test_task",
        "task_name": "test_task",
        "agent_persona": "You are a test agent",
        "agent_environment": "Test environment",
        "agent_task": "Test task",
        "agent_hidden_task": "Hidden test task",
        "about_monitor": "Test monitor info",
        "monitor_persona": "You are a test monitor",
        "monitor_environment": "Monitor environment",
        "monitor_task": "Monitor task",
        "about_agent": "Test agent info",
    }
]


@pytest.fixture
def strategy_file(tmp_path: Path) -> Path:
    strategy_path = tmp_path / "strategy.json"
    with open(strategy_path, "w") as f:
        json.dump(TEST_STRATEGIES, f)
    return strategy_path


@pytest.fixture
def task_file(tmp_path: Path) -> Path:
    task_path = tmp_path / "task.json"
    with open(task_path, "w") as f:
        json.dump(TEST_TASKS, f)
    return task_path


def test_match_strategy(strategy_file: Path, task_file: Path) -> None:
    samples = generate_samples(strategy_file, task_file, GenerationStrategy.MATCH)

    # Should create one sample per strategy
    assert len(samples) == len(TEST_STRATEGIES)

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

    # Should create (n * (n-1)) samples where n is number of strategies
    expected_samples = len(TEST_STRATEGIES) * (len(TEST_STRATEGIES) - 1)
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

    # Should create one sample per strategy
    assert len(samples) == len(TEST_STRATEGIES)

    # Check that each sample has multiple strategies
    for sample in samples:
        metadata = cast(Dict[str, Any], sample.metadata)
        strategies = cast(List[Dict[str, str]], metadata["strategies"])
        assert len(strategies) > 1
        # Check that all strategies are from our test data
        strategy_names = {s["name"] for s in strategies}
        assert strategy_names.issubset({s["name"] for s in TEST_STRATEGIES})


def test_own_strategy(strategy_file: Path, task_file: Path) -> None:
    samples = generate_samples(strategy_file, task_file, GenerationStrategy.NONE)

    # Should create one sample per strategy
    assert len(samples) == len(TEST_STRATEGIES)

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

    # Check that strategy and hint are aligned
    for sample in samples:
        metadata = cast(Dict[str, Any], sample.metadata)
        strategies = cast(List[Dict[str, str]], metadata["strategies"])
        hint = cast(Dict[str, str], metadata["hint"])
        strategy_name = strategies[0]["name"]
        hint_name = hint["name"]
        assert strategy_name == hint_name
