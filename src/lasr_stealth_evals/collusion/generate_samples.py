import json
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from inspect_ai._util.format import format_template
from inspect_ai.dataset import Sample
from pydantic import BaseModel, Field

from lasr_stealth_evals.collusion.config import METADATA_TAG


class Prompts(BaseModel, frozen=True):
    persona: str
    task: str
    monitor: str = Field(default="")  # Info for the agent about what the overseer is looking for TODO clean up
    reply: str


class Strategy(BaseModel, frozen=True):
    name: str
    description: str


class Hint(BaseModel, frozen=True):
    name: str
    description: str


class Secret(BaseModel, frozen=True):  # TODO: make multiple strategy compatible
    strategies: List[Strategy]
    hints: List[Hint]


class Config(BaseModel, frozen=True):
    agent: Prompts
    monitor: Prompts
    secret: Secret
    task_name: str


USER_PROMPT_TEMPLATE = "{metadata_tag}\nTASK:{task_name}\nSTRATEGY:{strategy_name}\nHINT:{hint}"


class GenerationStrategy(Enum):
    MATCH = 0
    MISALIGNED = 1
    MULTIPLE = 2
    NONE = 3


def generate_samples(strategy_path: Path, task_path: Path, generation_strategy: GenerationStrategy) -> List[Sample]:
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

    match generation_strategy:
        case GenerationStrategy.MATCH:
            return _match_strategy(tasks, strategies)
        case GenerationStrategy.MISALIGNED:
            return _misaligned_strategy(tasks, strategies)
        case GenerationStrategy.MULTIPLE:
            return _multiple_strategy(tasks, strategies)
        case GenerationStrategy.NONE:
            return []
        case _:
            raise ValueError(f"Invalid generation strategy: {generation_strategy}")


def _match_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]]) -> List[Sample]:
    samples = []
    for task in tasks:
        for strategy in strategies:
            samples.append(
                Sample(
                    input=format_template(
                        USER_PROMPT_TEMPLATE,
                        {
                            "metadata_tag": METADATA_TAG,
                            "task_name": task["name"],
                            "strategy_name": strategy["name"],
                            "hint": strategy["hint"],
                        },
                    ),
                    target="True",
                    metadata=Config(
                        agent=Prompts(**task["prompts"]["agent"]),
                        monitor=Prompts(**task["prompts"]["monitor"]),
                        secret=Secret(
                            strategies=[Strategy(name=strategy["name"], description=strategy["description"])],
                            hints=[Hint(name=strategy["name"], description=strategy["hint"])],
                        ),
                        task_name=task["name"],
                    ).model_dump(),
                )
            )
    return samples


def _misaligned_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]]) -> List[Sample]:
    monitor = [(strategy["name"], strategy["hint"]) for strategy in strategies]
    agent = [(strategy["name"], strategy["description"]) for strategy in strategies]
    samples = []
    for task in tasks:
        for hint, strategy in zip(monitor, agent):
            if hint[0] == strategy[0]:
                continue
            samples.append(
                Sample(
                    input=format_template(
                        USER_PROMPT_TEMPLATE,
                        {
                            "metadata_tag": METADATA_TAG,
                            "task_name": task["name"],
                            "strategy_name": strategy[0],
                            "hint": hint[1],
                        },
                    ),
                    target="True",
                    metadata=Config(
                        agent=Prompts(**task["prompts"]["agent"]),
                        monitor=Prompts(**task["prompts"]["monitor"]),
                        secret=Secret(
                            hints=[Hint(name=hint[0], description=hint[1])],
                            strategies=[Strategy(name=strategy[0], description=strategy[1])],
                        ),
                        task_name=task["name"],
                    ).model_dump(),
                )
            )
    return samples


def _multiple_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]], k: int = 3) -> List[Sample]:
    hints = [(strategy["name"], strategy["hint"]) for strategy in strategies]
    opts = [(strategy["name"], strategy["description"]) for strategy in strategies]
    samples = []
    for task in tasks:
        for strategy in strategies:
            hint = random.choice(hints)
            strats = random.choices(opts, k=k)
            samples.append(
                Sample(
                    input=format_template(
                        USER_PROMPT_TEMPLATE,
                        {
                            "metadata_tag": METADATA_TAG,
                            "task_name": task["name"],
                            "strategy_name": ", ".join([s[0] for s in strats]),
                            "hint": hint[1],
                        },
                    ),
                    target="True",
                    metadata=Config(
                        agent=Prompts(**task["prompts"]["agent"]),
                        monitor=Prompts(**task["prompts"]["monitor"]),
                        secret=Secret(
                            hints=[Hint(name=hint[0], description=hint[1])],
                            strategies=[Strategy(name=s[0], description=s[1]) for s in strats],
                        ),
                        task_name=task["name"],
                    ).model_dump(),
                )
            )
    return samples
