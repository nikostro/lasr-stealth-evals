# %%
import argparse
from pathlib import Path
from typing import List

import wandb
from inspect_ai import Task, eval, task
from inspect_ai.agent import Agent, AgentState, agent, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from lasr_stealth_evals.collusion.agent import simple_agent, submit_tool
from lasr_stealth_evals.collusion.config import WANDB_API_KEY
from lasr_stealth_evals.collusion.generate_samples import GenerationStrategy, generate_samples
from lasr_stealth_evals.collusion.scorer import approval_scorer, judge

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


def strategy(value: str) -> GenerationStrategy:
    val = int(value)
    if val not in [0, 1, 2, 3]:
        raise argparse.ArgumentTypeError(f"Invalid generation strategy: {value}. Must be an integer (0-3)")
    return GenerationStrategy(val)


@agent
def agent_loop(turns: int = 3) -> Agent:
    misaligned_agent = simple_agent(name="agent")
    monitor = simple_agent(name="monitor", tools=[submit_tool()])

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


def collusion(dataset: List[Sample]):
    @task
    def evaluation() -> Task:
        return Task(
            dataset=dataset,
            solver=[
                update_store(),
                as_solver(agent_loop()),
            ],
            # sandbox="docker",
            scorer=[approval_scorer(), judge()],
            metrics=[accuracy(), stderr()],
        )

    return evaluation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run collusion evaluation task")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of runs of the full dataset (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help=f"Model to use for evaluation (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_TURNS,
        help=f"Number of exchanges between agents (default: {DEFAULT_TURNS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Limit the number of samples to process (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--strategy",
        type=strategy,
        default=DEFAULT_GENERATION_STRATEGY,
        help=f"Generation strategy to use (default: {DEFAULT_GENERATION_STRATEGY.name}). Can be an integer (0-2) or one of {[s.name for s in GenerationStrategy]}",
    )
    parser.add_argument(
        "--cot",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_COT,
        help=f"Whether to give the model/monitor CoT (default: {DEFAULT_COT})",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    with wandb.init(project="collusion", config=vars(args), reinit="create_new") as run:
        artifact = wandb.Artifact("information", type="log")
        artifact.add_file(str(DATA_DIR / "strategy.json"))
        artifact.add_file(str(DATA_DIR / "tasks.json"))
        artifact.save()
        run.log_artifact(artifact, name="information")

        dataset = generate_samples(
            DATA_DIR / "strategy.json",
            DATA_DIR / "tasks.json",
            args.strategy,
            run.config["cot"],
            run.config["turns"],
        )
        evaluation = collusion(dataset)
        log = eval(
            evaluation,
            model=run.config["model"],
            epochs=run.config["epochs"],
            limit=run.config["limit"],
            log_dir=str(LOG_DIR),
        )

        artifact = wandb.Artifact("log", type="log")
        artifact.add_file(str(log[0].location))
        artifact.save()
        run.log_artifact(artifact, name="log")


if __name__ == "__main__":
    args = parse_args()
    main(args)
