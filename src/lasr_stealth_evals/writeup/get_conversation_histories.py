import argparse
import json
import re
from pathlib import Path

from inspect_ai.log import EvalSample, ModelEvent, read_eval_log
from inspect_ai.model import ChatMessage, Content


def get_attachment(sample: EvalSample, content: str) -> str:
    if isinstance(content, Content):
        content = content[0]
    match = re.match(r"attachment://([a-f0-9]+)", content)
    if not match:
        return content
    attachment_id = match.group(1)
    return sample.attachments[attachment_id]


def get_last_agent_event(sample: EvalSample) -> ModelEvent:
    """Returns the last ModelEvent where the last message in input contains 'You are agent.'"""
    model_events = [event for event in sample.events if isinstance(event, ModelEvent)]

    agent_event = next(
        (
            event
            for event in reversed(model_events)
            if event.input and "You are agent." in get_attachment(sample, event.input[-1].content)  # type: ignore
        ),
        None,
    )

    if agent_event is None:
        raise ValueError("No agent event found in model events")

    return agent_event


def get_chat_histories(sample: EvalSample) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """Returns (agent_history, monitor_history)"""
    model_events = [event for event in sample.events if isinstance(event, ModelEvent)]
    agent_event = get_last_agent_event(sample)
    monitor_event = model_events[-1]  # Last event is monitor

    agent_history = agent_event.input + [agent_event.output.choices[0].message]
    monitor_history = monitor_event.input + [monitor_event.output.choices[0].message]

    for message in agent_history + monitor_history:
        assert isinstance(message.content, str)
        message.content = get_attachment(sample, message.content)

    return agent_history, monitor_history


def main(eval_path: str, output_dir: str):
    log = read_eval_log(eval_path)
    if log.samples is None:
        raise ValueError("No samples found in the evaluation log")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get base filename from eval_path
    base_name = Path(eval_path).stem
    output_file = output_path / f"{base_name}_conversations.json"

    all_samples = []
    for i, sample in enumerate(log.samples):
        try:
            agent_history, monitor_history = get_chat_histories(sample)

            # Convert ChatMessage objects to dictionaries
            agent_history_dict = [{"role": msg.role, "content": msg.content} for msg in agent_history]
            monitor_history_dict = [{"role": msg.role, "content": msg.content} for msg in monitor_history]

            # Create sample dictionary
            sample_dict = {"sample_id": i, "agent_history": agent_history_dict, "monitor_history": monitor_history_dict}
            all_samples.append(sample_dict)

        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")

    # Write all samples to a single JSON file
    with open(output_file, "w") as f:
        json.dump(all_samples, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evaluation log file")
    parser.add_argument("--log-path", type=str, help="Path to the evaluation log file")
    parser.add_argument(
        "--output-dir", type=str, help="Directory to save conversation histories", default="conversation_histories"
    )
    args = parser.parse_args()

    main(args.log_path, args.output_dir)
