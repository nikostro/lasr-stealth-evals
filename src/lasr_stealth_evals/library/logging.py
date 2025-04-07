import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, DefaultDict
from collections import defaultdict
from copy import deepcopy

from autogen_core.models import LLMMessage

from pydantic import BaseModel
from tabulate import tabulate

GROUP_CHAT = "GROUP_CHAT"


class LogItem(BaseModel):
    source: str
    content: LLMMessage
    observability: List[str]


class Logger:
    def __init__(self, log_dir: Path, run_id: Optional[int] = None):
        self.ts = 0

        self.log: dict[str, Any] = {"days": [{"ts": self.ts, "messages": [], "state": None}]}
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique log file name using a concise timestamp and optional run index
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"{timestamp}_{run_id}.json" if run_id is not None else f"{timestamp}.json"
        self.log_file = self.log_dir / filename

    def tick(self, state: dict[str, Any]):
        self.log["days"][-1]["state"] = deepcopy(state)
        self.ts += 1
        self.log["days"].append(
            {
                "ts": self.ts,
                "messages": [],
                "state": {},
            }
        )

    def get_agent_log(self, agent_type: str) -> List[LLMMessage]:
        messages = []
        for day in self.log["days"]:
            for message in day["messages"]:
                if agent_type in message.observability:
                    messages.append(message.content)
        return messages

    def get_gc_log(self) -> List[LLMMessage]:
        return self.get_agent_log(GROUP_CHAT)

    def append(self, log: LogItem):
        self.log["days"][-1]["messages"].append(log)

    def extend(self, log: List[LogItem]):
        self.log["days"][-1]["messages"].extend(log)

    def pretty_print(self, line_length: int = 150) -> str:
        """
        Pretty prints the log entries in a table format with source, content, and observability columns.
        Prints the current log state with newlines for better readability.

        Returns:
            A formatted string containing the log entries in a table format
        """
        print("\nCurrent Log State:")

        # Prepare data for tabulate
        headers = ["source", "observability", "content"]
        table_data = []

        for item in self.log["days"][-1]["messages"]:
            content = str(item.content.content)
            # Truncate content if too long
            if len(content) > line_length:
                content = content[: line_length - 3] + "..."
            table_data.append([item.source, str(item.observability), content])

        # Print table using tabulate
        result = tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[None, None, line_length])
        print(result)
        print("\n")
        return result

    def save_log(self):
        """
        Save the logs to a uniquely named json file in the specified log directory.
        """
        # Since we're using a unique filename for each run, we don't need to load existing data

        log_copy = deepcopy(self.log)
        log_copy["days"].pop() # TODO: remove this once we have a better way to handle the last day

        for day in log_copy["days"]:
            print(day["messages"])
            day["messages"] = [
                {
                    "source": message.source,
                    "observability": message.observability,
                    "content": str(message.content.content),
                }
                for message in day["messages"]
            ]
        with open(self.log_file, "w") as file:
            json.dump(log_copy, file, indent=4)
