from typing import List

from autogen_core import AgentId
from autogen_core.models import LLMMessage
from pydantic import BaseModel, ConfigDict
from logging import getLogger

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

    def pretty_print(self) -> str:
        """
        Pretty prints the log entries in a table format with source and content columns.
        Prints the current log state with newlines for better readability.

        Returns:
            A formatted string containing the log entries in a table format
        """
        print("\nCurrent Log State:")

        # Calculate column widths
        max_source_width = max(len(item.source) for item in self.log)
        max_content_width = max(len(str(item.content.content)) for item in self.log)

        # Add padding for borders
        max_source_width += 2
        max_content_width += 2

        # Create header
        header = f"|{'source':<{max_source_width}}|{'content':<{max_content_width}}|"
        separator = f"|{'-' * (max_source_width-1)}|{'-' * (max_content_width-1)}|"

        # Create rows
        rows = []
        for item in self.log:
            source = item.source
            content = str(item.content.content)
            row = f"|{source:<{max_source_width}}|{content:<{max_content_width}}|"
            rows.append(row)

        # Combine all parts and print
        result = "\n".join([header, separator] + rows)
        print(result)
        print("\n")
        return result
