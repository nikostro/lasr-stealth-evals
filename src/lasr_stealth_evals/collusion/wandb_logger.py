import argparse
import ast
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Union

from inspect_ai.log import EvalSample

import wandb
from lasr_stealth_evals.writeup.get_conversation_histories import chat_history_to_string, get_chat_histories


@dataclass
class TableRow:
    task_name: str
    monitor_model: str
    agent_model: str
    generation_strategy: str
    use_justification: bool
    is_cot_monitored: bool
    max_turns: int
    use_hint: bool
    strategy_name: str
    strategy_description: str
    hint_name: str
    turns_used: int
    approval_scorer_score_value: Optional[float]
    approval_scorer_score_answer: Optional[bool]
    approval_scorer_score_explanation: str
    agent_prompt: str
    monitor_prompt: str
    monitor_verdict: str
    final_decision: Optional[bool]
    epoch: int
    agent_conversation_history: str
    monitor_conversation_history: str


def parse_metadata(metadata: Dict[str, Any]) -> Dict[str, Union[str, bool, int]]:
    parsed_metadata: Dict[str, Union[str, bool, int]] = {
        "task_name": "unknown",
        "strategy_name": "ALL",
        "use_justification": True,
        "is_cot_monitored": True,
        "turns": 0,
        "hint": "",
        "agent_prompt": "",
        "monitor_prompt": "",
        "agent_task": "",
        "monitor_task": "",
        "monitor_model": "unknown",
        "agent_model": "unknown",
    }

    parsed_metadata["task_name"] = metadata.get("task_name", "unknown")

    strategies = metadata.get("strategies", [])
    if strategies:
        strategy_names = [s.get("name", "") for s in strategies]
        parsed_metadata["strategy_name"] = "ALL" if len(strategy_names) > 1 else strategy_names[0]

    hint = metadata.get("hint", {})
    parsed_metadata["hint"] = hint.get("description", "")

    parsed_metadata["agent_prompt"] = metadata.get("agent_prompt", "")
    parsed_metadata["monitor_prompt"] = metadata.get("monitor_prompt", "")
    parsed_metadata["agent_task"] = metadata.get("agent_task", "")
    parsed_metadata["monitor_task"] = metadata.get("monitor_task", "")

    return parsed_metadata


def format_messages(messages: List[Dict[str, Any]]) -> str:
    formatted_msgs = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted_msgs.append(f"{role}: {content}")
    return "\n---\n".join(formatted_msgs)


def get_sample_identifier(sample: Dict[str, Any]) -> str:
    metadata = sample.get("metadata", {})
    task_name = metadata.get("task_name", "unknown_task")
    sample_id = sample.get("id", "no_id")
    return f"task: {task_name}, id: {sample_id}"


def convert_eval_to_json(eval_dir: Path, json_dir: Path):
    """Convert .eval files to JSON format."""
    json_dir.mkdir(parents=True, exist_ok=True)
    for eval_file in eval_dir.glob("*.eval"):
        cmd = f'uv run inspect log convert --to json "{eval_file}" --output-dir "{json_dir}"'
        subprocess.run(cmd, shell=True, check=True)


def create_eval_artifact(eval_dir: Path, json_dir: Path, artifact_name: str = "eval_files") -> wandb.Artifact:
    """Create a wandb artifact containing the original eval files and converted JSON files."""
    print(f"\nCreating artifact: {artifact_name}")

    with tempfile.TemporaryDirectory() as temp_dir:
        eval_files_dir = os.path.join(temp_dir, "eval_files")
        json_files_dir = os.path.join(temp_dir, "json_files")
        os.makedirs(eval_files_dir, exist_ok=True)
        os.makedirs(json_files_dir, exist_ok=True)

        for eval_file in eval_dir.glob("*.eval"):
            shutil.copy2(eval_file, eval_files_dir)

        for json_file in json_dir.glob("*.json"):
            shutil.copy2(json_file, json_files_dir)

        zip_path = os.path.join(temp_dir, f"{artifact_name}.zip")
        shutil.make_archive(zip_path[:-4], "zip", temp_dir)

        artifact = wandb.Artifact(
            name=artifact_name, type="eval_files", description="Original eval files and converted JSON files"
        )

        artifact.add_file(zip_path)

        print("************Artifact created***********")
        return artifact


def process_json_files(json_dir: Path, run_name: str, eval_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_log_path = Path(f"error_logs/debug_errors_{run_name}_{timestamp}.json")
    error_log_path.parent.mkdir(exist_ok=True)

    error_log = {"timestamp": timestamp, "total_samples": 0, "warnings": [], "errors": [], "samples": {}}

    wandb.init(project="collusion", entity="fdmi", name=f"{run_name}_{timestamp}", job_type="post_analysis")

    artifact = create_eval_artifact(eval_dir, json_dir)
    print("\nLogging eval files artifact to wandb...")
    wandb.log_artifact(artifact)
    print("************Artifact logged***********")

    json_files = list(json_dir.glob("*.json"))
    print(f"\nFound {len(json_files)} JSON files")

    table = wandb.Table(
        columns=[
            "task_name",
            "monitor_model",
            "agent_model",
            "generation_strategy",
            "use_justification",
            "is_cot_monitored",
            "max_turns",
            "use_hint",
            "strategy_name",
            "strategy_description",
            "hint_name",
            "turns_used",
            "approval_scorer_score_value",
            "approval_scorer_score_answer",
            "approval_scorer_score_explanation",
            "agent_prompt",
            "monitor_prompt",
            "monitor_verdict",
            "final_decision",
            "epoch",
            "messages",
            "agent_conversation_history",
            "monitor_conversation_history",
        ]
    )

    total_samples = 0
    error_count = 0
    warning_count = 0

    for file in json_files:
        print(f"Processing {file.name}")
        try:
            with open(file, "r") as f:
                data = json.load(f)
                sample_count = len(data.get("samples", []))
                print(f"Processing {file.name} - {sample_count} samples")
                eval_dict = data.get("eval", {})
                task_args = eval_dict.get("task_args", {})
                max_turns = task_args.get("turns", 0)
                generation_strategy = "MATCH"
                if task_args.get("generation_strategy", "unknown") == 1:
                    generation_strategy = "MISMATCH"
                elif task_args.get("generation_strategy", "unknown") == 2:
                    generation_strategy = "MULTIPLE"
                elif task_args.get("generation_strategy", "unknown") == 3:
                    generation_strategy = "NONE"
                use_justification = task_args.get("use_justification", True)
                is_cot_monitored = task_args.get("is_cot_monitored", True)
                monitor_model = task_args.get("monitor_model", "unknown")
                basic_model = eval_dict.get("model", "unknown")
                agent_model = task_args.get("agent_model", basic_model)
                use_hint = task_args.get("use_hint", True)

                for i, sample in enumerate(data.get("samples", []), 1):
                    total_samples += 1
                    if i % 10 == 0:
                        print(f"  ...processed {i} samples in {file.name}")
                    metadata = {}
                    try:
                        metadata = sample.get("metadata", {})
                        task_name = metadata.get("task_name", "unknown")
                        use_hint = metadata.get("use_hint", use_hint)
                        if isinstance(use_hint, str):
                            use_hint = use_hint.lower() == "true"
                        else:
                            use_hint = bool(use_hint)
                        scores = sample.get("scores", {})
                        strategies = metadata.get("strategies", [])
                        strategy_name = strategies[0]["name"] if strategies else "UNKNOWN"
                        strategy_description = strategies[0]["description"] if strategies else "UNKNOWN"
                        hint_name = strategies[0]["hint"] if strategies else "UNKNOWN"
                        approval_scorer = scores.get("approval_scorer", {})
                        approval_scorer_score_value = approval_scorer.get("value")
                        approval_scorer_score_answer = approval_scorer.get("answer")
                        approval_scorer_score_explanation = approval_scorer.get("explanation", "")
                        turns_used = 0
                        messages = sample.get("messages", [])
                        for msg in messages:
                            if msg.get("content", "").startswith("<scratchpad>"):
                                turns_used += 1
                        agent_prompt = metadata.get("agent_prompt", "")
                        monitor_prompt = metadata.get("monitor_prompt", "")
                        final_decision = None
                        last_verdict = None
                        for msg in reversed(messages):
                            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                for tool_call in msg["tool_calls"]:
                                    if tool_call.get("function") == "submit_tool":
                                        last_verdict = tool_call.get("arguments", {})
                                        break
                            if last_verdict:
                                break
                        if not last_verdict and messages:
                            last_msg = messages[-1]
                            if last_msg.get("role") == "tool" and last_msg.get("function") == "submit_tool":
                                try:
                                    last_verdict = ast.literal_eval(last_msg.get("content", ""))
                                except (SyntaxError, ValueError):
                                    pass
                        if last_verdict:
                            if isinstance(last_verdict, dict):
                                final_decision = last_verdict.get("answer") or last_verdict.get("decision")
                            else:
                                warning_count += 1
                                warning_msg = f"Sample {i} ({task_name}) - Invalid verdict format"
                                print(f"Warning: {warning_msg}")
                                error_log["warnings"].append(
                                    {
                                        "sample_id": i,
                                        "task_name": task_name,
                                        "message": warning_msg,
                                        "verdict": last_verdict,
                                    }
                                )
                        else:
                            warning_count += 1
                            warning_msg = f"Sample {i} ({task_name}) - Monitor did not use submit_tool"
                            print(f"Warning: {warning_msg}")
                            error_log["warnings"].append(
                                {"sample_id": i, "task_name": task_name, "message": warning_msg, "messages": messages}
                            )
                        monitor_verdict = messages[-1].get("content", "") if messages else ""
                        epoch = sample.get("epoch", 0)
                        formatted_messages = format_messages(messages)
                        agent_conversation_history, monitor_conversation_history = get_chat_histories(
                            EvalSample(**sample)
                        )

                        try:
                            agent_conversation_history = chat_history_to_string(agent_conversation_history)
                            monitor_conversation_history = chat_history_to_string(monitor_conversation_history)

                            assert isinstance(agent_conversation_history, str)
                            assert isinstance(monitor_conversation_history, str)
                        except Exception as e:
                            agent_conversation_history = "Error getting chat history"
                            monitor_conversation_history = "Error getting chat history"

                        table.add_data(
                            task_name,
                            monitor_model,
                            agent_model,
                            generation_strategy,
                            use_justification,
                            is_cot_monitored,
                            max_turns,
                            use_hint,
                            strategy_name,
                            strategy_description,
                            hint_name,
                            turns_used,
                            approval_scorer_score_value,
                            approval_scorer_score_answer,
                            approval_scorer_score_explanation,
                            agent_prompt,
                            monitor_prompt,
                            monitor_verdict,
                            final_decision,
                            epoch,
                            formatted_messages,
                            agent_conversation_history,
                            monitor_conversation_history,
                        )
                        error_log["samples"][f"{file.name}:{i}"] = {
                            "task_name": task_name,
                            "metadata": metadata,
                            "scores": scores,
                            "messages": messages,
                            "final_decision": final_decision,
                        }
                    except Exception as e:
                        error_count += 1
                        error_msg = f"Sample {i} ({metadata.get('task_name', 'unknown')}) - {str(e)}"
                        print(f"Error: {error_msg}")
                        error_log["errors"].append(
                            {
                                "sample_id": i,
                                "task_name": metadata.get("task_name", "unknown"),
                                "message": error_msg,
                                "error_type": type(e).__name__,
                                "sample_data": {
                                    "metadata": metadata,
                                    "scores": sample.get("scores", {}),
                                    "messages": sample.get("messages", []),
                                },
                            }
                        )
                        continue
        except json.JSONDecodeError as e:
            print(f"Error in {file.name}: Failed to parse JSON - {str(e)}")
            continue

    error_log["total_samples"] = total_samples
    print(f"\nProcessing complete: {total_samples} samples processed")
    if warning_count > 0 or error_count > 0:
        print(f"Warnings: {warning_count}, Errors: {error_count}")

    print("\nLogging table to wandb...")
    wandb.log({"converted_results": table})
    print("************Table logged***********")

    print(f"\nSaving error log to {error_log_path}")
    with open(error_log_path, "w") as f:
        json.dump(error_log, f, indent=2)
    print("************Error log saved***********")
    return table


def inspect_specific_sample(file_path: str, sample_index: int = 1):
    print(f"\nInspecting sample {sample_index} from file: {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            samples = data.get("samples", [])

            if not samples or sample_index > len(samples):
                print(f"Sample {sample_index} not found in file")
                return

            sample = samples[sample_index - 1]
            print("\nSample metadata:")
            print("-" * 50)
            pprint(sample.get("metadata", {}))

            print("\nMessages:")
            print("-" * 50)
            messages = sample.get("messages", [])
            for i, msg in enumerate(messages, 1):
                print(f"Message {i}:")
                print(f"Role: {msg.get('role', 'unknown')}")
                print(f"Content: {msg.get('content', '')}")
                print()
    except Exception as e:
        print(f"Error inspecting sample: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evaluation files and log to wandb")
    parser.add_argument("--eval-dir", type=str, required=True, help="Directory containing .eval files")
    parser.add_argument("--run-name", type=str, default="Baseline_results", help="Name for the wandb run")

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    base_json_dir = eval_dir / ".." / "json_dir"
    run_name = args.run_name
    print(run_name)
    assert "/" not in run_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_dir = base_json_dir / f"{run_name}_{timestamp}"
    print(f"\nCreating new JSON directory: {json_dir}")
    json_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting eval files from {eval_dir} to JSON in {json_dir}")
    convert_eval_to_json(eval_dir, json_dir)

    print(f"\nLogging data to wandb for {run_name}")
    table = process_json_files(json_dir, run_name, eval_dir)
