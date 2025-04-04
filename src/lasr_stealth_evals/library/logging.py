from typing import List, Optional
from datetime import datetime

from autogen_core.models import LLMMessage
from pydantic import BaseModel
from tabulate import tabulate
import json
from pathlib import Path

GROUP_CHAT = "GROUP_CHAT"


class LogItem(BaseModel):
    source: str
    content: LLMMessage
    observability: List[str]


class Logger:
    def __init__(self, log_dir: Path, filename: Optional[str] = None):
        self.log: List[LogItem] = []
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique log file name using a concise timestamp
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        if not filename:
            filename = f"{timestamp}.json"
        self.log_file = self.log_dir / filename

    def get_agent_log(self, agent_type: str) -> List[LLMMessage]:
        return [item.content for item in self.log if agent_type in item.observability]

    def get_gc_log(self) -> List[LLMMessage]:
        return [item.content for item in self.log if GROUP_CHAT in item.observability]

    def append(self, log: LogItem):
        self.extend([log])

    def extend(self, log: List[LogItem]):
        self.log.extend(log)

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

        for item in self.log:
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
        table_data = []

        for item in self.log:
            content = str(item.content.content)
            table_data.append({"source": item.source, "observability": str(item.observability), "content": content})

        with open(self.log_file, "w") as file:
            json.dump(table_data, file, indent=4)
