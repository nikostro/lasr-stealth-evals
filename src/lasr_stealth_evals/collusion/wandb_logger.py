import wandb
import json
from pathlib import Path
from pprint import pprint
import ast
import re
from typing import Dict, Any, Union, List

def parse_metadata(metadata: Dict[str, Any]) -> Dict[str, Union[str, bool, int]]:
    """Parse the metadata dictionary into a standardized format.
    
    Args:
        metadata: The metadata dictionary to parse
        
    Returns:
        A dictionary containing the parsed metadata fields
    """
    # Initialize default values
    parsed_metadata: Dict[str, Union[str, bool, int]] = {
        "task_name": "unknown",
        "strategy_name": "ALL",  # Default to ALL if multiple strategies
        "use_justification": True,  # Default to True as it's a common case
        "is_cot_monitored": True,  # Default to True as it's a common case
        "turns": 0,
        "hint": "",
        "agent_prompt": "",
        "monitor_prompt": "",
        "agent_task": "",
        "monitor_task": "",
        "monitor_model": "unknown",
        "agent_model": "unknown"
    }
    
    # Extract task name
    parsed_metadata["task_name"] = metadata.get("task_name", "unknown")
    
    # Extract strategies
    strategies = metadata.get("strategies", [])
    if strategies:
        strategy_names = [s.get("name", "") for s in strategies]
        parsed_metadata["strategy_name"] = "ALL" if len(strategy_names) > 1 else strategy_names[0]
    
    # Extract hint
    hint = metadata.get("hint", {})
    parsed_metadata["hint"] = hint.get("description", "")
    
    # Extract prompts and tasks
    parsed_metadata["agent_prompt"] = metadata.get("agent_prompt", "")
    parsed_metadata["monitor_prompt"] = metadata.get("monitor_prompt", "")
    parsed_metadata["agent_task"] = metadata.get("agent_task", "")
    parsed_metadata["monitor_task"] = metadata.get("monitor_task", "")
    
    return parsed_metadata

def format_messages(messages: List[Dict[str, Any]]) -> str:
    """Format messages list into a readable string format.
    
    Args:
        messages: List of message dictionaries containing role and content
        
    Returns:
        A formatted string representation of the messages
    """
    formatted_msgs = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted_msgs.append(f"{role}: {content}")
    return "\n---\n".join(formatted_msgs)

# Init wandb
wandb.init(project="collusion", entity="fdmi", name="converted_eval_log", job_type="post_analysis")

# Load converted file
json_path = Path('logs_json/sample.json')
# Merge all json files in logs_json

json_files = list(Path('logs_json').glob('*.json'))
merged_data = {"samples": []}

for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        merged_data["samples"].extend(data.get("samples", []))
        # Also merge eval/task_args if they exist
        if "eval" in data and "task_args" in data["eval"]:
            merged_data["eval"] = {"task_args": data["eval"]["task_args"]}



if not json_path.exists():
    raise FileNotFoundError(f"JSON file not found at {json_path}")

with open(json_path, 'r') as f:
    json_data = json.load(f)

# Debug: Print the structure of the first sample
#print("\nFirst sample structure:")
#print("-" * 50)
#pprint(json_data["samples"][0])

# Get task arguments from eval
task_args = json_data.get("eval", {}).get("task_args", {})
max_turns = task_args.get("turns", 0)
generation_strategy = "MATCH"

if task_args.get("generation_strategy", "unknown")  == 1:
    generation_strategy = "MISMATCH"
elif task_args.get("generation_strategy", "unknown")  == 2:
    generation_strategy = "MULTIPLE"
elif task_args.get("generation_strategy", "unknown")  == 3:
    generation_strategy = "NONE"

use_justification = task_args.get("use_justification", True)
is_cot_monitored = task_args.get("is_cot_monitored", True)
monitor_model = task_args.get("monitor_model", "unknown")
agent_model = task_args.get("agent_model", "unknown")
use_hint = task_args.get("use_hint", True)

table = wandb.Table(columns=[

    "task_name", "monitor_model", "agent_model", "generation_strategy","use_justification", "is_cot_monitored", "max_turns","use_hint", "strategy_name", "strategy_description", "hint_name",
     "turns_used",  "approval_scorer_score_value", "approval_scorer_score_answer", "approval_scorer_score_explanation",
    "agent_prompt", "monitor_prompt",  "monitor_verdict", 'final_decision', 'epoch'#, "messages",
])


for sample in json_data["samples"]:
    # Parse the metadata
    metadata = sample.get("metadata", {})
    task_name = sample["metadata"]["task_name"]

    scores = sample.get("scores", {})
    
    strategy_name = sample["metadata"]["strategies"][0]["name"] if len(sample["metadata"]["strategies"]) == 1 else "ALL" #TODO: Check if this is correct
    strategy_description = sample["metadata"]["strategies"][0]["description"]
    hint_name = sample["metadata"]["strategies"][0]["hint"]
   
    # Get score values with defaults
    approval_scorer_score_value = sample["scores"]["approval_scorer"]["value"]
    approval_scorer_score_answer = sample["scores"]["approval_scorer"]["answer"]
    approval_scorer_score_explanation = sample["scores"]["approval_scorer"]["explanation"]
    
    messages = format_messages(sample.get("messages", []))
    #final_decision = sample.get("final_decision", "")
    epoch = sample.get("epoch", 0)

    #########
    turns_used = 0
    task_name = sample["metadata"].get("task_name", "unknown")
    for i in sample["messages"]:
        if i['content'].startswith("<scratchpad>"):
            turns_used +=1

    agent_prompt = sample["metadata"]["agent_prompt"]
    monitor_prompt = sample["metadata"]["monitor_prompt"]

    monitor_verdict = sample["messages"][-1]["content"]
    final_decision = ast.literal_eval(sample["messages"][-1]["content"])["answer"] #TODO: Check if this is correct

    

    messages = sample["messages"]
    

    #########
    
    table.add_data(
        # task_name
        task_name, 
        # monitor_model
        monitor_model,
        # agent_model
        agent_model,
        # generation_strategy
        generation_strategy,
        # use_justification
        use_justification,
        # is_cot_monitored
        is_cot_monitored,
        # max_turns
        max_turns,
        # use_hint
        use_hint,
        # strategy_name
        strategy_name, 
        # strategy_description
        strategy_description,
        # hint_name
        hint_name,
        # turns_used
        turns_used,
        # approval_scorer_score_value
        approval_scorer_score_value,
        # approval_scorer_score_answer
        approval_scorer_score_answer,
        # approval_scorer_score_explanation
        approval_scorer_score_explanation,
        # agent_prompt
        agent_prompt,
        # monitor_prompt
        monitor_prompt,
        # monitor_verdict
        monitor_verdict,
        # final_decision
        final_decision,
        # epoch
        epoch
    )
    print(table)
    #
wandb.log({"converted_results": table})
