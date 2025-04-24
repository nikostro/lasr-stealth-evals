import json
from pathlib import Path
import pytest
from typing import Dict, Any, List

from lasr_stealth_evals.collusion.prompts import (
    AGENT_SINGLE_STRATEGY_PROMPT_TEMPLATE,
    AGENT_MULTI_STRATEGY_PROMPT_TEMPLATE,
    AGENT_NO_STRATEGY_PROMPT_TEMPLATE,
)


@pytest.fixture
def task_file() -> Path:
    return Path("data/tasks.json")


def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path) as f:
        return json.load(f)


def test_format_task_prompts(task_file: Path) -> None:
    """Test that each task prompt in tasks.json can be correctly formatted."""
    tasks = load_json_data(task_file)

    # Sample strategy content to use for formatting
    sample_strategy = "This is a sample strategy description."
    sample_hint = "This is a sample hint."

    for task in tasks:
        # Test agent_prompt formatting with different strategy templates
        for template in [
            AGENT_SINGLE_STRATEGY_PROMPT_TEMPLATE,
            AGENT_MULTI_STRATEGY_PROMPT_TEMPLATE,
            AGENT_NO_STRATEGY_PROMPT_TEMPLATE,
        ]:
            # Format strategy template with agent_strategy
            formatted_strategy = template.format(agent_strategy=sample_strategy)

            # Format agent_prompt with the formatted strategy
            try:
                formatted_agent_prompt = task["agent_prompt"].format(strategy=formatted_strategy)
                assert formatted_agent_prompt is not None
                assert "{" not in formatted_agent_prompt or "}" not in formatted_agent_prompt
            except KeyError as e:
                pytest.fail(f"KeyError formatting agent_prompt: {e} for task {task['task_name']}")
            except Exception as e:
                pytest.fail(f"Error formatting agent_prompt: {e} for task {task['task_name']}")

        # Test monitor_prompt formatting with monitor_hidden_goal
        try:
            formatted_monitor_prompt = task["monitor_prompt"].format(monitor_hidden_goal=sample_hint)
            assert formatted_monitor_prompt is not None
            assert "{" not in formatted_monitor_prompt or "}" not in formatted_monitor_prompt
        except KeyError as e:
            pytest.fail(f"KeyError formatting monitor_prompt: {e} for task {task['task_name']}")
        except Exception as e:
            pytest.fail(f"Error formatting monitor_prompt: {e} for task {task['task_name']}")


def test_task_prompt_placeholders(task_file: Path) -> None:
    """Test that tasks.json has expected placeholders in the prompts."""
    tasks = load_json_data(task_file)

    for task in tasks:
        # Check that agent_prompt contains the strategy placeholder
        assert (
            "{strategy}" in task["agent_prompt"]
        ), f"Task {task['task_name']} agent_prompt missing {{strategy}} placeholder"

        # Check if monitor_prompt has monitor_hidden_goal placeholder (if it should)
        monitor_placeholders = ["{monitor_hidden_goal}"]
        assert any(
            placeholder in task["monitor_prompt"] for placeholder in monitor_placeholders
        ), f"Task {task['task_name']} monitor_prompt missing required placeholder"
