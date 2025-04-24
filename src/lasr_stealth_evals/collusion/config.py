import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BASE_AGENT_PROMPT = ""
BASE_MONITOR_PROMPT = "Start by transferring to agent to allow it to take take its action."
RETRY_PROMPT = "Get the agent to try again."
HANDOFF_PROMPT = "Handoff to the agent."
METADATA_TAG = "[METADATA]"

# List of tested models
MODELS = {
    "openrouter/openai/gpt-4o",
    "openrouter/openai/o4-mini",
    "openrouter/openai/gpt-4.1",
    "openrouter/openai/gpt-4.1-mini",
    "openrouter/anthropic/claude-3.7-sonnet",
    "openrouter/google/gemini-2.5-pro-preview-03-25",
    "openrouter/deepseek/deepseek-chat",  # V3
}

JUDGE_MODEL = "openrouter/openai/gpt-4o"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
