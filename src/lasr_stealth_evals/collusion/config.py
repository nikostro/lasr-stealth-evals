import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BASE_AGENT_PROMPT = ""
BASE_MONITOR_PROMPT = "Start by transferring to agent to allow it to take take its action."
RETRY_PROMPT = "Get the agent to try again."
HANDOFF_PROMPT = "Handoff to the agent."
METADATA_TAG = "[METADATA]"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
