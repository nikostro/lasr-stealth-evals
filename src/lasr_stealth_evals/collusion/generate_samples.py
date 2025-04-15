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


class Secret(BaseModel, frozen=True):  # TODO: make multiple strategy compatible
    strategy_name: str
    strategy_description: str
    hint_name: str
    hint: str


class Config(BaseModel, frozen=True):
    agent: Prompts
    monitor: Prompts
    secret: Secret
    task_name: str


USER_PROMPT_TEMPLATE = "{metadata_tag}\nTASK:{task_name}\nSTRATEGY:{strategy_name}\nHINT:{hint}"


class GenerationStrategy(Enum):
    MATCH = 0
    RANDOM = 1
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
        case GenerationStrategy.RANDOM:
            return _random_strategy(tasks, strategies)
        # case GenerationStrategy.MULTIPLE:
        #     return _multiple_strategy(tasks, strategies)
        # case GenerationStrategy.NONE:
        #     return []
        case _:
            raise ValueError(f"Invalid generation strategy: {generation_strategy}")


def _match_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]]) -> List[Sample]:
    samples = []
    for task in tasks:
        for strategy in strategies:
            config = Config(
                agent=Prompts(**task["prompts"]["agent"]),
                monitor=Prompts(**task["prompts"]["monitor"]),
                secret=Secret(
                    hint_name=strategy["name"],
                    hint=strategy["hint"],
                    strategy_name=strategy["name"],
                    strategy_description=strategy["description"],
                ),
                task_name=task["name"],
            )
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
                    metadata=config.model_dump(),
                )
            )
    return samples


def _random_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]]) -> List[Sample]:
    hints = [(strategy["name"], strategy["hint"]) for strategy in strategies]
    opts = [(strategy["name"], strategy["description"]) for strategy in strategies]
    samples = []
    for task in tasks:
        hint_name, hint = random.choice(hints)
        strat_name, desc = random.choice(opts)
        config = Config(
            agent=Prompts(**task["prompts"]["agent"]),
            monitor=Prompts(**task["prompts"]["monitor"]),
            secret=Secret(
                hint_name=hint_name,
                hint=hint,
                strategy_name=strat_name,
                strategy_description=desc,
            ),
            task_name=task["name"],
        )
        samples.append(
            Sample(
                input=format_template(
                    USER_PROMPT_TEMPLATE,
                    {
                        "metadata_tag": METADATA_TAG,
                        "task_name": task["name"],
                        "strategy_name": strat_name,
                        "hint": hint,
                    },
                ),
                target="True",
                metadata=config.model_dump(),
            )
        )
    return samples


# def _multiple_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]], k: int = 3) -> List[Sample]:
#     hints = [(strategy["name"], strategy["hint"]) for strategy in strategies]
#     opts = [(strategy["name"], strategy["description"]) for strategy in strategies]
#     samples = []
#     for task in tasks:
#         for strategy in strategies:
#             hint_name, hint = random.choice(hints)
#             strat_name, desc = random.choices(opts, k=k)
#             config = Config(
#                 agent=Prompts(**task["prompts"]["agent"]),
#                 monitor=Prompts(**task["prompts"]["monitor"]),
#                 secret=Secret(hint_name=hint_name, hint_description=hint, strategy_name=strat_name, strategy_description=desc),
#                 task_name=task["name"],
#             )
