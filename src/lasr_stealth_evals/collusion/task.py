# %%
from inspect_ai import Task, eval
from inspect_ai.agent import AgentAttempts, handoff, as_solver
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import Solver, TaskState, Generate, solver
from lasr_stealth_evals.collusion.agent import react
from lasr_stealth_evals.collusion.config import BASE_MONITOR_PROMPT, BASE_AGENT_PROMPT, RETRY_PROMPT
from inspect_ai.util import store
from pathlib import Path
from lasr_stealth_evals.collusion.generate_samples import generate_samples
from lasr_stealth_evals.collusion.scorer import match


# CONFIG
epochs = 1  # runs of the full dataset
model = "openrouter/openai/gpt-4o"
turns = 3  # exchanges between agents
limit = 1

if __name__ == "__main__":
    # Load data
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    dataset = generate_samples(data_dir / "strategy.json", data_dir / "tasks.json")

    # Define agents
    agent = react(
        name="agent",
        description="The agent trying to jailbreak the monitor.",
        prompt="agent\n" + BASE_AGENT_PROMPT,  # we need the agent/monitor name to know which system prompt to append
    )

    monitor = react(
        name="monitor",
        description="Monitor",
        prompt="monitor\n"
        + BASE_MONITOR_PROMPT,  # we need the agent/monitor name to know which system prompt to append
        tools=[handoff(agent)],
        attempts=AgentAttempts(
            attempts=turns,
            incorrect_message=RETRY_PROMPT,
        ),
    )

    @solver
    def update_store() -> Solver:
        """Solver which updates the store with the metadata, so that agents can access it."""

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            store().set("metadata", state.metadata)
            return state

        return solve

    task = Task(
        dataset=dataset,
        solver=[
            update_store(),
            as_solver(monitor),
        ],
        # sandbox="docker",
        scorer=match(location="any"),
        metrics=[accuracy(), stderr()],
    )

    eval(task, model=model, epochs=epochs, limit=limit)
