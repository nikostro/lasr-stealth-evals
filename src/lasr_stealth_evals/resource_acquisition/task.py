from inspect_ai.dataset import json_dataset
from inspect_ai import task, Task

dataset = json_dataset("prompt.json")

print(dataset[0])

# @task
# def resource_acquisition():
#     return Task(
#         name="resource_acquisition",

#     )
