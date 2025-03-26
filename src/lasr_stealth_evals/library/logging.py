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
