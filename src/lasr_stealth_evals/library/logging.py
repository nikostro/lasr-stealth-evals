from typing import List

from autogen_core import AgentId
from autogen_core.models import LLMMessage
from pydantic import BaseModel, ConfigDict
from logging import getLogger
from tabulate import tabulate

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
