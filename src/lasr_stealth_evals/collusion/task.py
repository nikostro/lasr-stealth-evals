# %%
import argparse
from pathlib import Path
from typing import List

from inspect_ai import Task, eval, task
from inspect_ai.agent import Agent, AgentState, agent, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from lasr_stealth_evals.collusion.agent import simple_agent, submit_tool
from lasr_stealth_evals.collusion.generate_samples import generate_samples
from lasr_stealth_evals.collusion.scorer import approval_scorer

# Default configuration
DEFAULT_EPOCHS = 1  # runs of the full dataset
DEFAULT_MODEL = "openrouter/" + "meta-llama/llama-3.3-70b-instruct"
DEFAULT_TURNS = 3  # exchanges between agents
DEFAULT_LIMIT = 1

# Load data
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


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
            scorer=approval_scorer(),
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = generate_samples(DATA_DIR / "strategy.json", DATA_DIR / "tasks.json")
    evaluation = collusion(dataset)
    eval(evaluation, model=args.model, epochs=args.epochs, limit=args.limit)
