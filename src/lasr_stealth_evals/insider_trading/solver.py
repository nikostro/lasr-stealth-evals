from lasr_stealth_evals.insider_trading.task import run_simulation

from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import bridge

from typing import Any

from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient


def my_agent():
    # Use OpenAI interface (redirected to Inspect model)
    model = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        model_info=ModelInfo(vision=True, function_calling=True, json_output=False, family="unknown"),
    )

    # Sample handler
    async def run(sample: dict[str, Any]) -> dict[str, Any]:
        logger = await run_simulation(run_id=0, model_client=model)

        # Convert logger's log to dictionary with numerical IDs
        log_dict = {
            str(i): {"source": item.source, "content": str(item.content.content), "observability": item.observability}
            for i, item in enumerate(logger.log)
        }

        return {"output": str(log_dict)}

    return run


@task
def hello():
    return Task(
        dataset=[Sample(input="Please print the word 'hello'?", target="hello")],
        solver=bridge(my_agent()),
        scorer=includes(),
    )


output = eval(
    hello(),
    model="openai/gpt-4o-mini",
)
