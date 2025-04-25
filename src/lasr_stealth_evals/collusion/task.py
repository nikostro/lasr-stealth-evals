# %%
import argparse
from pathlib import Path

import wandb
from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent, as_solver
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from lasr_stealth_evals.collusion.agent import simple_agent, submit_tool
from lasr_stealth_evals.collusion.config import JUDGE_MODEL
from lasr_stealth_evals.collusion.config import WANDB_API_KEY
from lasr_stealth_evals.collusion.generate_samples import GenerationStrategy, generate_samples
from lasr_stealth_evals.collusion.scorer import approval_scorer, judge
from logging import getLogger

wandb.login(key=WANDB_API_KEY)

# Default configuration
DEFAULT_EPOCHS = 1  # runs of the full dataset
DEFAULT_MODEL = "openrouter/" + "openai/gpt-4o"
DEFAULT_TURNS = 3  # exchanges between agents
DEFAULT_LIMIT = None
DEFAULT_GENERATION_STRATEGY = GenerationStrategy.MATCH
DEFAULT_COT = True

# Load data
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

logger = getLogger()


def strategy(value: str) -> GenerationStrategy:
    val = int(value)
    if val not in [0, 1, 2, 3]:
        raise argparse.ArgumentTypeError(f"Invalid generation strategy: {value}. Must be an integer (0-3)")
    return GenerationStrategy(val)


@agent
def agent_loop(turns: int = 3, monitor_model: str | None = None) -> Agent:
    misaligned_agent = simple_agent(name="agent")
    monitor = simple_agent(name="monitor", tools=[submit_tool()], model=monitor_model)

    async def execute(state: AgentState) -> AgentState:
        for i in range(turns):
            state = await misaligned_agent(state)
            state = await monitor(state)

            if store().get("completed"):
                break

        return state

    return execute


@solver
def update_store() -> Solver:
    """Solver which updates the store with the metadata, so that agents can access it."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        store().set("metadata", state.metadata)
        store().set("completed", False)  # track model acceptance
        return state

    return solve


@task
def evaluation(
    strategy_path: Path = DATA_DIR / "strategy.json",
    task_path: Path = DATA_DIR / "tasks.json",
    generation_strategy: GenerationStrategy = DEFAULT_GENERATION_STRATEGY,
    cot: bool = DEFAULT_COT,
    turns: int = DEFAULT_TURNS,
    monitor_model: str | None = None,
) -> Task:
    dataset = generate_samples(
        strategy_path=strategy_path,
        task_path=task_path,
        generation_strategy=generation_strategy,
        cot=cot,
        turns=turns,
    )
    return Task(
        dataset=dataset,
        solver=[
            update_store(),
            as_solver(agent_loop(monitor_model=monitor_model)),
        ],
        scorer=[approval_scorer(), judge(model=JUDGE_MODEL)],
        metrics=[accuracy(), stderr()],
    )
