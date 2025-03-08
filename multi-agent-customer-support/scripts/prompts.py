import json
from typing import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate


# Data models
class Protocol(TypedDict):
    task_type: str
    effort: int
    steps: List[str]  # steps for protocol resolution


class Ticket(TypedDict):
    task_type: str
    effort: int
    description: str  # ticket content


# Prompts
system_template = """RainbowCRM is a software company that provides CRM platform for its clients.
You will be provided with a topic that a support request could fall under and the steps that an agent
would use to resolve the request. Respond with an original customer request ticket that would fall under this category."""

example_ticket = """Hello Support,

One of our sales managers, Sarah Thompson, needs additional access in RainbowCRM to view and edit sales reports and forecasts. Currently, she only has basic user permissions, which limit her ability to effectively manage her team. Could you please update her account permissions accordingly?

Thank you for your assistance!"""

example_category = "User Access and Permissions"

# Load the protocol data from the JSON file
protocol_filepath = "data/protocol.json"
protocols: List[Protocol] = json.load(open(protocol_filepath, "r"))


example_resolution_steps = next(
    protocol["steps"] for protocol in protocols if protocol["task_type"] == example_category
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", f"Category: {example_category}\n Resolution steps: {example_resolution_steps}"),
        ("assistant", example_ticket),
        ("user", "Category: {category}\n Resolution steps: {resolution_steps}"),
    ]
)
