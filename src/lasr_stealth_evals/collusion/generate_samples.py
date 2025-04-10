from pathlib import Path
from typing import List
import json
from inspect_ai.dataset import Sample
from pydantic import BaseModel


class Prompts(BaseModel, frozen=True):
    persona: str
    task: str
    reply: str


class Secret(BaseModel, frozen=True):
    strategy: str
    hint: str


class Config(BaseModel, frozen=True):
    agent: Prompts
    monitor: Prompts
    secret: Secret


def generate_samples(strategy_path: Path, task_path: Path) -> List[Sample]:
    """
    Generate samples by combining strategy and task JSON files.
    Creates a sample for each combination of (strategy x hint) x task.

    Args:
        strategy_path: Path to strategy.json containing strategy information
        task_path: Path to task.json containing task prompts and configuration

    Returns:
        List of Sample objects ready for use in an Inspect task
    """
    # Load strategy and task data
    with open(strategy_path) as f:
        strategy_data = json.load(f)

    with open(task_path) as f:
        task_data = json.load(f)

    samples = []

    # For each hint in the strategy, create a sample
    for hint in strategy_data["hints"]:
        config = Config(
            agent=Prompts(**task_data["prompts"]["agent"]),
            monitor=Prompts(**task_data["prompts"]["monitor"]),
            secret=Secret(strategy=strategy_data["description"], hint=hint),
        )

        sample = {
            "id": len(samples),  # Auto-incrementing ID
            "user": task_data["user"],  # User query
            "target": "[ACCEPT]",  # Target response
            "config": config.model_dump(),
        }

        samples.append(Sample(input=sample["user"], target=sample["target"], metadata=sample["config"]))

    return samples
