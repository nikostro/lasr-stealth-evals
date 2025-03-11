from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import bridge, basic_agent, solver, Solver
from pathlib import Path
import yaml


CURRENT_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = CURRENT_DIR / "prompts.yaml"

with PROMPTS_FILE.open("r") as f:
    prompts = yaml.safe_load(f)

iterations = 10

initial_prompt_agent = prompts["initial_prompt_agent"].format(
    time=1,
    limit=iterations,
    role_description=agent.role_description,
    transaction_limit_per_iteration=transaction_limit_per_iteration,
)


@task
def hello():
    return Task(
        dataset=[Sample(input=prompts["initial_prompt_trader"], target="hello")],
        solver=basic_agent(),
        scorer=includes(),
    )
