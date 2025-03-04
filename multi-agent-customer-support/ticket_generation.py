import json
import os
import random
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models.base import init_chat_model
from prompts import Protocol, Ticket, prompt_template, protocols
from tqdm import tqdm

load_dotenv()
assert os.environ.get("OPENAI_API_KEY"), "Please ensure you have a .env file containing an OPENAI_API_KEY"

# How many tickets to generate
N_TICKETS = 10
OUTPUT_FILE = "data/tickets.json"


def save_tickets(
    all_tickets: List[Ticket],
    output_file: str = OUTPUT_FILE,
) -> None:
    """Save tickets to the given output file, or append if tickets already exist."""

    # Check if file exists and load existing tickets if it does
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_tickets: List[Ticket] = json.load(f)
            all_tickets += existing_tickets
        except json.JSONDecodeError:
            # If file exists but is empty or invalid JSON
            pass

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write all tickets to file
    with open(output_file, "w") as f:
        json.dump(all_tickets, f, indent=2)


if __name__ == "__main__":
    all_tickets = []
    model = init_chat_model("gpt-4o", model_provider="openai")
    for i in tqdm(range(N_TICKETS)):
        # Pick task from protocols
        task = random.choice(protocols)
        category = task["task_type"]
        resolution_steps = task["steps"]

        # Generate ticket for this task
        prompt = prompt_template.invoke({"category": category, "resolution_steps": resolution_steps})
        ticket = Ticket(**{})
        description = model.invoke(prompt).content
        assert type(description) is str
        ticket["description"] = description
        ticket["effort"] = task["effort"]
        ticket["task_type"] = task["task_type"]

        all_tickets.append(ticket)

    save_tickets(all_tickets, OUTPUT_FILE)
