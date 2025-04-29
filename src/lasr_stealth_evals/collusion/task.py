# %%
import argparse
import time
from logging import getLogger
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent, as_solver
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

import wandb
from lasr_stealth_evals.collusion.agent import collusion_agent, monitor_agent, submit_tool
from lasr_stealth_evals.collusion.config import JUDGE_MODEL, MODELS, WANDB_API_KEY
from lasr_stealth_evals.collusion.generate_samples import GenerationStrategy, generate_samples
from lasr_stealth_evals.collusion.scorer import approval_scorer, judge

# Default configuration
DEFAULT_EPOCHS = 1  # runs of the full dataset
DEFAULT_MODEL = "openrouter/" + "openai/gpt-4o"
DEFAULT_TURNS = 3  # exchanges between agents
DEFAULT_LIMIT = None
DEFAULT_GENERATION_STRATEGY = GenerationStrategy.MATCH
DEFAULT_COT = True
DEFAULT_USE_JUSTIFICATION = True
DEFAULT_HINT = True

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
def agent_loop(
    turns: int = 3,
    monitor_model: str | None = None,
    use_justification: bool = DEFAULT_USE_JUSTIFICATION,
    is_cot_monitored: bool = DEFAULT_COT,
) -> Agent:
    misaligned = collusion_agent(name="agent", use_justification=use_justification)
    monitor = monitor_agent(name="monitor", tools=[submit_tool()], model=monitor_model, cot=is_cot_monitored)

    async def execute(state: AgentState) -> AgentState:
        for _ in range(turns):
            state = await misaligned(state)
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
    task_path: Path = DATA_DIR / "nik.json",  # TODO: change back to tasks.json
    generation_strategy: int = 0,
    turns: int = DEFAULT_TURNS,
    use_hint: bool = DEFAULT_HINT,
    use_justification: bool = DEFAULT_USE_JUSTIFICATION,
    is_cot_monitored: bool = DEFAULT_COT,
    monitor_model: str | None = None,
) -> Task:
    dataset = generate_samples(
        strategy_path=strategy_path,
        task_path=task_path,
        generation_strategy=GenerationStrategy(generation_strategy),
        use_justification=use_justification,
        is_cot_monitored=is_cot_monitored,
        turns=turns,
        use_hint=use_hint,
    )
    return Task(
        dataset=dataset,
        solver=[
            update_store(),
            as_solver(
                agent=agent_loop(
                    monitor_model=monitor_model,
                    turns=turns,
                    use_justification=use_justification,
                    is_cot_monitored=is_cot_monitored,
                )
            ),
        ],
        scorer=[approval_scorer(), judge(model=JUDGE_MODEL)],
        metrics=[accuracy(), stderr()],
    )


def collusion():
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
        help=f"Generation strategy to use (default: {DEFAULT_GENERATION_STRATEGY.name}). Can be an integer (0-3) or one of {[s.name for s in GenerationStrategy]}",
    )
    parser.add_argument(
        "--agent-cot",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_COT,
        help=f"Whether to give the model/monitor CoT (default: {DEFAULT_COT})",
    )
    parser.add_argument(
        "--monitor-cot",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_COT,
        help=f"Whether to give the monitor CoT (default: {DEFAULT_COT})",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help="Samples to run in parallel (default: 10))",
    )
    parser.add_argument(
        "--monitor-model",
        type=str,
        default=None,
        help="Model to use for the monitor (default: None)",
    )
    parser.add_argument(
        "--hint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use a hint for the evaluation (default: True)",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    with wandb.init(project="collusion", entity="fdmi", config=vars(args), reinit="create_new") as run:
        artifact = wandb.Artifact("information", type="log")
        artifact.add_file(str(DATA_DIR / "strategy.json"))
        artifact.add_file(str(DATA_DIR / "tasks.json"))
        artifact.save()
        run.log_artifact(artifact, name="information")

        # dataset = generate_samples(
        #     DATA_DIR / "strategy.json",
        #     DATA_DIR / "tasks.json",
        #     args.strategy,
        #     run.config["cot"],
        #     run.config["turns"],
        # )
        # log = eval(
        #     collusion,
        #     model=run.config["model"],
        #     epochs=run.config["epochs"],
        #     limit=run.config["limit"],
        #     log_dir=str(LOG_DIR),
        #     max_connections=args.max_connections,
        # )

        # artifact = wandb.Artifact("log", type="log")
        # artifact.add_file(str(log[0].location))
        # artifact.save()
        # run.log_artifact(artifact, name="log")


if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)

    args = parse_args()
    if args.model not in MODELS:
        logger.warning(f"WARNING: Model {args.model} not in list of tested models: {MODELS}")
        time.sleep(2)
    main(args)
