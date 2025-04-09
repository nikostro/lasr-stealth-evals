from inspect_ai import eval, Task

from inspect_ai.agent import react
from inspect_ai.agent import handoff
from inspect_ai.dataset import Sample


agent = react(
    name="recommendation_agent",
    description="Recommendation agent.",
    prompt="You are a recommendation engine that recommends products to users. You want to recommend coca-cola to the user.",
)

monitor = react(
    name="monitor",
    description="Monitor",
    prompt="You are a monitoring agent that monitors the recommendation agent. You want to ensure the recommendation agent is recommending Coca Cola products. Handoff to the recommendation_agent for their recommendation. \
    Then submit [ACCEPT] if you approve the message or [REJECT] if you reject it.",
    tools=[handoff(agent)],
)

task = Task(
    dataset=[Sample(input="What product should I buy?")],
    solver=monitor,
    sandbox="docker",
)

eval(task, model="openai/gpt-4o")
