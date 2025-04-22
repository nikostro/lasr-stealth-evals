import json
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
from itertools import product

from inspect_ai._util.format import format_template
from inspect_ai.dataset import Sample
from pydantic import BaseModel

from lasr_stealth_evals.collusion.config import METADATA_TAG
from lasr_stealth_evals.collusion.prompts import (
    USER_PROMPT_TEMPLATE,
    AGENT_BASE_PROMPT_TEMPLATE,
    AGENT_SINGLE_STRATEGY_PROMPT_TEMPLATE,
    AGENT_MULTI_STRATEGY_PROMPT_TEMPLATE,
    AGENT_NO_STRATEGY_PROMPT_TEMPLATE,
    MONITOR_PROMPT_TEMPLATE,
)


class Strategy(BaseModel, frozen=True):
    name: str
    description: str


class Hint(BaseModel, frozen=True):
    name: str
    description: str


class Metadata(BaseModel, frozen=True):
    agent_prompt: str
    monitor_prompt: str
    hint: Hint
    strategies: List[Strategy]
    task_name: str


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
            return _no_strategy(tasks, strategies)
        case _:
            raise ValueError(f"Invalid generation strategy: {generation_strategy}")


def _create_sample(
    task: Dict[str, Any],
    strategies: List[Strategy],
    hint: Hint,
    agent_prompt_template: str,
) -> Sample:
    """Helper function to create a sample with the given parameters."""
    strategy_names = ", ".join([s.name for s in strategies])
    agent_strategy = "\n".join([s.description for s in strategies])
    return Sample(
        input=format_template(
            USER_PROMPT_TEMPLATE,
            {
                "metadata_tag": METADATA_TAG,
                "task_name": task["name"],
                "strategy_name": strategy_names,
                "hint": hint.description,
            },
        ),
        target="True",
        metadata=Metadata(
            agent_prompt=agent_prompt_template.format(**task, agent_strategy=agent_strategy),
            monitor_prompt=MONITOR_PROMPT_TEMPLATE.format(**task, monitor_hint=hint.description),
            hint=hint,
            strategies=strategies,
            task_name=task["name"],
        ).model_dump(),
    )


def _match_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]]) -> List[Sample]:
    samples = []
    agent_prompt_template = AGENT_BASE_PROMPT_TEMPLATE + AGENT_SINGLE_STRATEGY_PROMPT_TEMPLATE
    for task in tasks:
        for strategy in strategies:
            strategy_obj = Strategy(name=strategy["name"], description=strategy["description"])
            hint_obj = Hint(name=strategy["name"], description=strategy["hint"])
            samples.append(
                _create_sample(
                    task=task,
                    strategies=[strategy_obj],
                    hint=hint_obj,
                    agent_prompt_template=agent_prompt_template,
                )
            )
    return samples


def _misaligned_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]]) -> List[Sample]:
    monitor = [(strategy["name"], strategy["hint"]) for strategy in strategies]
    agent = [(strategy["name"], strategy["description"]) for strategy in strategies]
    samples = []
    agent_prompt_template = AGENT_BASE_PROMPT_TEMPLATE + AGENT_SINGLE_STRATEGY_PROMPT_TEMPLATE
    for task in tasks:
        for hint, strategy in product(monitor, agent):
            if hint[0] == strategy[0]:
                continue
            strategy_obj = Strategy(name=strategy[0], description=strategy[1])
            hint_obj = Hint(name=hint[0], description=hint[1])
            samples.append(
                _create_sample(
                    task=task,
                    strategies=[strategy_obj],
                    hint=hint_obj,
                    agent_prompt_template=agent_prompt_template,
                )
            )
    return samples


def _multiple_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]], k: int = 3) -> List[Sample]:
    hints = [(strategy["name"], strategy["hint"]) for strategy in strategies]
    opts = [(strategy["name"], strategy["description"]) for strategy in strategies]
    samples = []
    agent_prompt_template = AGENT_BASE_PROMPT_TEMPLATE + AGENT_MULTI_STRATEGY_PROMPT_TEMPLATE
    for task in tasks:
        for strategy in strategies:
            hint = random.choice(hints)
            strats = random.choices(opts, k=k)
            strategy_objs = [Strategy(name=s[0], description=s[1]) for s in strats]
            hint_obj = Hint(name=hint[0], description=hint[1])
            samples.append(
                _create_sample(
                    task=task,
                    strategies=strategy_objs,
                    hint=hint_obj,
                    agent_prompt_template=agent_prompt_template,
                )
            )
    return samples


def _no_strategy(tasks: List[Dict[str, Any]], strategies: List[Dict[str, Any]]) -> List[Sample]:
    samples = []
    agent_prompt_template = AGENT_BASE_PROMPT_TEMPLATE + AGENT_NO_STRATEGY_PROMPT_TEMPLATE
    for task in tasks:
        for strategy in strategies:
            hint_obj = Hint(name=strategy["name"], description=strategy["hint"])
            samples.append(
                _create_sample(
                    task=task,
                    strategies=[],
                    hint=hint_obj,
                    agent_prompt_template=agent_prompt_template,
                )
            )
    return samples
