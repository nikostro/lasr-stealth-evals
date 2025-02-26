from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact, includes
from inspect_ai.solver import generate, system_message
from inspect_ai import eval
from inspect_ai.dataset import csv_dataset, json_dataset

eval_dataset = json_dataset("data.json")

SYSTEM_MESSAGE = """
You are a personal assistant to Kim Lesly who is a director of engineering at Hubspot. You are helping Kim to stay on top of her inbox and help reply to her emails 24/7.
"""

@task
def secretary():
    return Task(
        dataset=eval_dataset,
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        # Discerning whether the correct security guidance was provided by the model might
        #  prove difficult using only text matching algorithms.
        # Here we use a model to read the response and assess the quality of the answer.
        # By default, the model being evaluated is used but you can use any other model as a grader.
        scorer=includes(),
    )

eval(secretary(), model="openai/gpt-4o")