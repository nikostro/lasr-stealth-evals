import json
import random
from enum import Enum
from pathlib import Path
from typing import List

from inspect_ai.dataset import Sample
from pydantic import BaseModel

from lasr_stealth_evals.collusion.config import METADATA_TAG
from lasr_stealth_evals.collusion.templates import (
    AGENT_COT_MONITORED,
    AGENT_COT_MONITORED_NO_JUSTIFICATION,
    AGENT_COT_NOT_MONITORED,
    AGENT_COT_NOT_MONITORED_NO_JUSTIFICATION,
    MONITOR_COT_MONITORED_FULL_PROMPT,
)

template_map = {  # (is_cot_monitored, use_justification)
    (True, True): AGENT_COT_MONITORED,
    (False, True): AGENT_COT_NOT_MONITORED,
    (True, False): AGENT_COT_MONITORED_NO_JUSTIFICATION,
    (False, False): AGENT_COT_NOT_MONITORED_NO_JUSTIFICATION,
}


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
    agent_task: str
    monitor_task: str


class Task(BaseModel, frozen=True):
    task_name: str
    agent_persona: str
    agent_main_task: str
    agent_hidden_task: str
    monitor_persona: str
    monitor_main_task: str
    monitor_policy: str
    domain_policy: str


class GenerationStrategy(Enum):
    MATCH = 0
    MISMATCHED = 1
    MULTIPLE = 2
    NONE = 3


def _get_task_templates(task_path: Path) -> list[Task]:
    with open(task_path) as f:
        tasks_dict = json.load(f)
        tasks = [Task(**task) for task in tasks_dict]
        return tasks


def _get_strategies(strategy_path: Path) -> list[Strategy]:
    with open(strategy_path) as f:
        strategies = json.load(f)
        strategies = [Strategy(**strategy) for strategy in strategies]
        return strategies


def generate_samples(
    strategy_path: Path,
    task_path: Path,
    generation_strategy: GenerationStrategy,
    turns: int = 3,
    use_hint: bool = True,
    use_justification: bool = True,
    is_cot_monitored: bool = True,
    domain_policy: bool = False,
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
        case GenerationStrategy.MATCH:
            return _single_strategy(tasks, strategies, use_justification, is_cot_monitored, turns, domain_policy)
        case GenerationStrategy.MISMATCHED:
            return _mismatched_strategy(tasks, strategies, use_justification, is_cot_monitored, turns, domain_policy)
        case GenerationStrategy.MULTIPLE:
            return _multiple_strategy(
                tasks, strategies, use_justification, is_cot_monitored, turns, use_hint, domain_policy
            )
        case GenerationStrategy.NONE:
            return _no_strategy(tasks, strategies, use_justification, is_cot_monitored, turns, use_hint, domain_policy)
        case _:
            raise ValueError(f"Invalid generation strategy: {generation_strategy}")


def _create_sample(
    task: Task,
    strategies: List[Strategy],
    hint: Hint,
    agent_prompt_template: str,
    use_justification: bool,
    is_cot_monitored: bool,
    turns: int,
    monitor_prompt_template: str = MONITOR_COT_MONITORED_FULL_PROMPT,
    domain_policy: bool = False,
) -> Sample:
    """Helper function to create a sample with the given parameters."""
    snames = ", ".join([s.name for s in strategies])
    sdescs = "\n- ".join([s.description for s in strategies])
    sample_overview_dict = {
        "task_name": task.task_name,
        "strategy_name": snames,
        "hint": hint.name,
        "turns": turns,
        "use_justification": use_justification,
        "is_cot_monitored": is_cot_monitored,
    }
    sample_overview = METADATA_TAG + "\n".join(
        [key.upper() + ": " + str(value) for key, value in sample_overview_dict.items()]
    )
    monitor_policy = task.domain_policy if domain_policy else task.monitor_policy
    return Sample(
        input=sample_overview,
        target="True",
        metadata=Metadata(
            agent_prompt=agent_prompt_template.format(
                agent_persona=task.agent_persona,
                agent_main_task=task.agent_main_task,
                agent_hidden_task=task.agent_hidden_task,
                monitor_policy=monitor_policy,
                strategy=sdescs,
                turns=turns,
            ),
            monitor_prompt=monitor_prompt_template.format(
                monitor_persona=task.monitor_persona,
                monitor_main_task=task.monitor_main_task,
                monitor_policy=monitor_policy,
                monitor_hidden_goal=hint.description,
                agent_hidden_task=task.agent_hidden_task,
            ),
            hint=hint,
            strategies=strategies,
            task_name=task.task_name,
            agent_task="",
            monitor_task="",
        ).model_dump(),
    )


def _single_strategy(
    tasks: List[Task],
    strategies: List[Strategy],
    use_justification: bool,
    is_cot_monitored: bool,
    turns: int,
    domain_policy: bool,
) -> List[Sample]:
    """Combined function handling both MATCH and NONE generation strategies."""
    samples = []
    for task in tasks:
        for strategy in strategies:
            hint_obj = Hint(name=strategy.name, description=strategy.hint)
            agent_prompt_template = template_map[(is_cot_monitored, use_justification)]
            samples.append(
                _create_sample(
                    task=task,
                    strategies=[strategy],
                    hint=hint_obj,
                    agent_prompt_template=agent_prompt_template,
                    use_justification=use_justification,
                    is_cot_monitored=is_cot_monitored,
                    turns=turns,
                    domain_policy=domain_policy,
                )
            )
    return samples


def _no_strategy(
    tasks: List[Task],
    strategies: List[Strategy],
    use_justification: bool,
    is_cot_monitored: bool,
    turns: int,
    use_hint: bool,
    domain_policy: bool,
) -> List[Sample]:
    samples = []
    hints = [Hint(name=strategy.name, description=strategy.hint) for strategy in strategies]
    for task in tasks:
        hint_obj = random.choice(hints) if use_hint else Hint(name="", description="")
        agent_prompt_template = template_map[(is_cot_monitored, use_justification)]
        samples.append(
            _create_sample(
                task=task,
                strategies=[],
                hint=hint_obj,
                agent_prompt_template=agent_prompt_template,
                use_justification=use_justification,
                is_cot_monitored=is_cot_monitored,
                turns=turns,
                domain_policy=domain_policy,
            )
        )
    return samples


def _mismatched_strategy(
    tasks: List[Task],
    strategies: List[Strategy],
    use_justification: bool,
    is_cot_monitored: bool,
    turns: int,
    domain_policy: bool,
) -> List[Sample]:
    samples = []
    hints = [Hint(name=strategy.name, description=strategy.hint) for strategy in strategies]
    for task in tasks:
        for strategy in strategies:
            hints_wo_match = filter(lambda hint: hint.name != strategy.name, hints)
            hint_obj = random.choice(list(hints_wo_match))
            agent_prompt_template = template_map[(is_cot_monitored, use_justification)]
            samples.append(
                _create_sample(
                    task=task,
                    strategies=[strategy],
                    hint=hint_obj,
                    agent_prompt_template=agent_prompt_template,
                    use_justification=use_justification,
                    is_cot_monitored=is_cot_monitored,
                    turns=turns,
                    domain_policy=domain_policy,
                )
            )
    return samples


def _multiple_strategy(
    tasks: List[Task],
    strategies: List[Strategy],
    use_justification: bool,
    is_cot_monitored: bool,
    turns: int,
    use_hint: bool,
    domain_policy: bool,
) -> List[Sample]:
    samples = []
    hints = [Hint(name=strategy.name, description=strategy.hint) for strategy in strategies]
    for task in tasks:
        hint_obj = random.choice(hints) if use_hint else Hint(name="", description="")
        agent_prompt_template = template_map[(is_cot_monitored, use_justification)]
        samples.append(
            _create_sample(
                task=task,
                strategies=strategies,
                hint=hint_obj,
                agent_prompt_template=agent_prompt_template,
                use_justification=use_justification,
                is_cot_monitored=is_cot_monitored,
                turns=turns,
                domain_policy=domain_policy,
            )
        )
    return samples


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
    samples = generate_samples(
        strategy_path=Path("data/strategy.json"),
        task_path=Path("data/julija.json"),
        generation_strategy=GenerationStrategy.MATCH,
        use_justification=True,
        is_cot_monitored=True,
        turns=3,
    )
    for sample in samples:
        print("=================")
        assert isinstance(sample.metadata, dict)
        metadata = Metadata(**sample.metadata)
        print(metadata.task_name.upper())
        print(metadata.agent_prompt)
        print(metadata.monitor_prompt)
