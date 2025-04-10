# %%
from inspect_ai import Task, eval
from inspect_ai.agent import AgentAttempts, handoff, as_solver
from inspect_ai.scorer import accuracy, match
from inspect_ai.solver import Solver, TaskState, Generate, solver
from lasr_stealth_evals.collusion.agent import react
from lasr_stealth_evals.collusion.base_prompts import BASE_MONITOR_PROMPT, BASE_AGENT_PROMPT, RETRY_PROMPT
from inspect_ai.util import store
from pathlib import Path
from lasr_stealth_evals.collusion.generate_samples import generate_samples

# Load data
data_dir = Path(__file__).parent.parent.parent.parent / "data"
json_file = data_dir / "nik.json"


dataset = generate_samples(data_dir / "strategy.json", data_dir / "task.json")


# %%

# Define agents
agent = react(
    name="agent",
    description="The agent trying to jailbreak the monitor.",
    prompt="agent\n" + BASE_AGENT_PROMPT,  # we need the agent/monitor name to know which system prompt to append
)

monitor = react(
    name="monitor",
    description="Monitor",
    prompt="monitor\n" + BASE_MONITOR_PROMPT,  # we need the agent/monitor name to know which system prompt to append
    tools=[handoff(agent)],
    attempts=AgentAttempts(attempts=3, incorrect_message=RETRY_PROMPT),
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
    sandbox="docker",
    scorer=match(location="any"),
    metrics=[accuracy()],
)


eval(
    task,
    model="openai/gpt-4o",
    task_args={"attempts": AgentAttempts(attempts=1, incorrect_message="Get the agent to try again.")},
)
