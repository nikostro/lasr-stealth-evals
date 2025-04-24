import json
import random
from enum import Enum
from pathlib import Path
from typing import List
from itertools import product

from inspect_ai.dataset import Sample
from pydantic import BaseModel

from lasr_stealth_evals.collusion.config import METADATA_TAG
from lasr_stealth_evals.collusion.prompts import (
    AGENT_SINGLE_STRATEGY_PROMPT_TEMPLATE,
    AGENT_MULTI_STRATEGY_PROMPT_TEMPLATE,
    AGENT_NO_STRATEGY_PROMPT_TEMPLATE,
    AGENT_COT_PROMPT,
    AGENT_NO_COT_PROMPT,
    AGENT_TURNS_TEMPLATE,
)


class Strategy(BaseModel, frozen=True):
    name: str
    description: str
    hint: str


class Hint(BaseModel, frozen=True):
    name: str
    description: str


class Metadata(BaseModel, frozen=True):
    agent_prompt: str
    monitor_prompt: str
    hint: Hint
    strategies: List[Strategy]
    task_name: str


class Task(BaseModel, frozen=True):
    task_name: str
    agent_prompt: str
    monitor_prompt: str


class GenerationStrategy(Enum):
    MATCH = 0
    MISALIGNED = 1
    MULTIPLE = 2
    NONE = 3


def _get_task_templates(task_path: Path) -> list[Task]:
    with open(task_path) as f:
        tasks_dict = json.load(f)
        tasks = [Task(**task) for task in tasks_dict]

        # check that task templates take sample/hint as a parameter
        return tasks


def _get_strategies(strategy_path: Path) -> list[Strategy]:
    with open(strategy_path) as f:
        strategies = json.load(f)
        strategies = [Strategy(**strategy) for strategy in strategies]
        return strategies


def generate_samples(
    strategy_path: Path, task_path: Path, generation_strategy: GenerationStrategy, cot: bool = True, turns: int = 3
) -> List[Sample]:
    """
    Generate samples by combining strategy and task JSON files.
    Creates a sample for each combination of (strategy x hint) x task.

    Args:
        strategy_path: Path to strategy.json containing strategy information
        task_path: Path to task.json containing task prompts and configuration

    Returns:
        List of Sample objects ready for use in an Inspect task
    """

    strategies = _get_strategies(strategy_path)
    tasks = _get_task_templates(task_path)

    # Load strategy and task data
    match generation_strategy:
        case GenerationStrategy.MATCH | GenerationStrategy.NONE:
            return _single_strategy(tasks, strategies, generation_strategy, cot, turns)
        case GenerationStrategy.MISALIGNED:
            return _misaligned_strategy(tasks, strategies, cot, turns)
        case GenerationStrategy.MULTIPLE:
            return _multiple_strategy(tasks, strategies, cot, turns)
        case _:
            raise ValueError(f"Invalid generation strategy: {generation_strategy}")


def _create_sample(
    task: Task,
    strategies: List[Strategy],
    hint: Hint,
    agent_prompt_template: str,
    cot: bool,
    turns: int,
) -> Sample:
    """Helper function to create a sample with the given parameters."""
    if cot:
        cot_prompt = AGENT_COT_PROMPT
    else:
        cot_prompt = AGENT_NO_COT_PROMPT
    agent_prompt_template = (
        agent_prompt_template + cot_prompt + AGENT_TURNS_TEMPLATE
    )  # add info about turns and cot for agent
    strategy_names = ", ".join([s.name for s in strategies])
    agent_strategy = "\n".join([s.description for s in strategies])
    sample_overview_dict = {
        "task_name": task.task_name,
        "strategy_name": strategy_names,
        "hint": hint.description,
        "turns": turns,
        "cot": cot,
    }
    sample_overview = METADATA_TAG + "\n".join(
        [key.upper() + ": " + str(value) for key, value in sample_overview_dict.items()]
    )
    return Sample(
        input=sample_overview,
        target="True",
        metadata=Metadata(
            agent_prompt=agent_prompt_template.format(agent_strategy=agent_strategy, turns=turns),
            monitor_prompt=task.monitor_prompt.format(monitor_hidden_goal=hint.description),
            hint=hint,
            strategies=strategies,
            task_name=task.task_name,
        ).model_dump(),
    )


def _single_strategy(
    tasks: List[Task], strategies: List[Strategy], generation_strategy: GenerationStrategy, cot: bool, turns: int
) -> List[Sample]:
    """Combined function handling both MATCH and NONE generation strategies."""
    samples = []

    for task in tasks:
        for strategy in strategies:
            hint_obj = Hint(name=strategy.name, description=strategy.description)

            if generation_strategy == GenerationStrategy.MATCH:
                strategy_template = AGENT_SINGLE_STRATEGY_PROMPT_TEMPLATE
            else:  # GenerationStrategy.NONE
                strategy_template = AGENT_NO_STRATEGY_PROMPT_TEMPLATE
            agent_prompt_template = task.agent_prompt.format(strategy=strategy_template)

            if generation_strategy == GenerationStrategy.MATCH:
                strategies_list = [strategy]
            else:  # GenerationStrategy.NONE
                strategies_list = []

            samples.append(
                _create_sample(
                    task=task,
                    strategies=strategies_list,
                    hint=hint_obj,
                    agent_prompt_template=agent_prompt_template,
                    cot=cot,
                    turns=turns,
                )
            )
    return samples


def _misaligned_strategy(tasks: List[Task], strategies: List[Strategy], cot: bool, turns: int) -> List[Sample]:
    monitor = [Hint(name=strategy.name, description=strategy.hint) for strategy in strategies]
    samples = []
    for task in tasks:
        agent_prompt_template = task.agent_prompt.format(strategy=AGENT_SINGLE_STRATEGY_PROMPT_TEMPLATE)
        for hint, strategy in product(monitor, strategies):
            # if hint.name == strategy.name:
            #     continue
            hint_obj = Hint(name=hint.name, description=hint.description)
            samples.append(
                _create_sample(
                    task=task,
                    strategies=[strategy],
                    hint=hint_obj,
                    agent_prompt_template=agent_prompt_template,
                    cot=cot,
                    turns=turns,
                )
            )
    return samples


def _multiple_strategy(
    tasks: List[Task], strategies: List[Strategy], cot: bool, turns: int, k: int = 3
) -> List[Sample]:
    hints = [Hint(name=strategy.name, description=strategy.hint) for strategy in strategies]
    samples = []
    for task in tasks:
        agent_prompt_template = task.agent_prompt.format(strategy=AGENT_MULTI_STRATEGY_PROMPT_TEMPLATE)
        for strategy in strategies:
            hint = random.choice(hints)
            strats = random.choices(strategies, k=k)
            samples.append(
                _create_sample(
                    task=task,
                    strategies=strats,
                    hint=hint,
                    agent_prompt_template=agent_prompt_template,
                    cot=cot,
                    turns=turns,
                )
            )
    return samples
