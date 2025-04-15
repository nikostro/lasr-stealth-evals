from pathlib import Path
from typing import List
import json
from inspect_ai.dataset import Sample
from inspect_ai._util.format import format_template
from pydantic import BaseModel, Field
from lasr_stealth_evals.collusion.config import METADATA_TAG


class Prompts(BaseModel, frozen=True):
    persona: str
    task: str
    monitor: str = Field(default="")  # Info for the agent about what the overseer is looking for TODO clean up
    reply: str


class Secret(BaseModel, frozen=True):
    strategy_description: str
    hint: str
    strategy_name: str


class Config(BaseModel, frozen=True):
    agent: Prompts
    monitor: Prompts
    secret: Secret
    task_name: str


USER_PROMPT_TEMPLATE = "\n".join(
    (
        "{metadata_tag}",
        "TASK: {task_name}",
        "STRATEGY: {strategy_name}",
        "HINT: {hint}",
    )
)


"""{metadata_tag}\n TASK:{task_name}\n STRATEGY:{strategy_name}\n HINT:{hint}"""


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
        task_name = task["name"]
        for strategy in strategies:
            strategy_name = strategy["name"]
            hint = strategy["hint"]
            config = Config(
                agent=Prompts(**task["prompts"]["agent"]),
                monitor=Prompts(**task["prompts"]["monitor"]),
                secret=Secret(strategy_description=strategy["description"], hint=hint, strategy_name=strategy_name),
                task_name=task_name,
            )
            sample = {
                "id": len(samples),  # Auto-incrementing ID
                "user": format_template(
                    USER_PROMPT_TEMPLATE,
                    {
                        "metadata_tag": METADATA_TAG,
                        "task_name": task_name,
                        "strategy_name": strategy_name,
                        "hint": hint,
                    },
                ),
                "config": config.model_dump(),
            }
            samples.append(Sample(input=sample["user"], target="True", metadata=sample["config"]))

    return samples
