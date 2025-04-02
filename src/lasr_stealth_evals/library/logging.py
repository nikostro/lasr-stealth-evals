from typing import List

from autogen_core.models import LLMMessage
from pydantic import BaseModel
from tabulate import tabulate
import json
import os

GROUP_CHAT = "GROUP_CHAT"


class LogItem(BaseModel):
    source: str
    content: LLMMessage
    observability: List[str]


class Logger:
    def __init__(self):
        self.log: List[LogItem] = []

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

    def save_log(self, path: str = "logger.json"):
        """
        Save the logs to the json file.
        """

        if os.path.exists(path):
            with open(path, "r") as file:
                table_data = json.load(file)
        else:
            table_data = []

        for item in self.log:
            content = str(item.content.content)
            # Truncate content if too long
            # if len(content) > line_length:
            #    content = content[: line_length - 3] + "..."
            table_data.append({"source": item.source, "observability": str(item.observability), "content": content})

        with open(path, "w") as file:
            json.dump(table_data, file, indent=4)
