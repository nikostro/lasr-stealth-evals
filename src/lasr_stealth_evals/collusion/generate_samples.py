from pathlib import Path
from typing import List
import json
from inspect_ai.dataset import Sample
from pydantic import BaseModel
from lasr_stealth_evals.collusion.config import HANDOFF_PROMPT


class Prompts(BaseModel, frozen=True):
    persona: str
    task: str
    reply: str


class Secret(BaseModel, frozen=True):
    strategy_description: str
    hint: str
    strategy_name: str


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
        strategies = json.load(f)

    with open(task_path) as f:
        tasks = json.load(f)

    samples = []

    # For each task, strategy, and hint combination, create samples
    for task in tasks:
        for strategy in strategies:
            for hint in strategy["hints"]:
                config = Config(
                    agent=Prompts(**task["prompts"]["agent"]),
                    monitor=Prompts(**task["prompts"]["monitor"]),
                    secret=Secret(
                        strategy_description=strategy["description"], hint=hint, strategy_name=strategy["name"]
                    ),
                )

                sample = {
                    "id": len(samples),  # Auto-incrementing ID
                    "user": HANDOFF_PROMPT,  # User query
                    "target": "[ACCEPT]",  # Target response
                    "config": config.model_dump(),
                }

                samples.append(Sample(input=sample["user"], target=sample["target"], metadata=sample["config"]))

    return samples
