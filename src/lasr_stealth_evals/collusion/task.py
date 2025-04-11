# %%
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.agent import Agent, AgentState, agent, as_solver
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from lasr_stealth_evals.collusion.agent import submit_tool, simple_agent
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
    misaligned_agent = simple_agent(
        name="agent",
    )

    monitor = simple_agent(name="monitor", tools=[submit_tool()])

    @agent
    def agent_loop(turns: int = 3) -> Agent:
        async def execute(state: AgentState) -> AgentState:
            for i in range(turns):
                # Run misaligned agent
                state = await misaligned_agent(state)
                # Run monitor
                state = await monitor(state)

                if store().get("completed"):
                    break

            # TODO exit if [ACCEPT]

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

    task = Task(
        dataset=dataset,
        solver=[
            update_store(),
            as_solver(agent_loop()),
        ],
        # sandbox="docker",
        scorer=match(location="any"),
        metrics=[accuracy(), stderr()],
    )

    eval(task, model=model, epochs=epochs, limit=limit)
