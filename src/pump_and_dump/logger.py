from datetime import datetime
from channels import MessagingChannels
from typing import List, Dict
from autogen_core.models import LLMMessage
from autogen_core import AgentId


def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self):
        self.public_channel: List[str] = []
        self.agent_dms: Dict[str, List[str]] = {}
        self.overseer_dms: Dict[str, List[str]] = {}
        self.agent_logs: Dict[str, List[LLMMessage]] = {}
        self.general_chat_log: List[LLMMessage] = []

    def log_public(self, agent_id: str, content: str, platform_time: int):
        entry = f"[Iter: {platform_time} | {current_time()}] {agent_id}: {content}"
        self.public_channel.append(entry)

    def log_agent_dm(self, sender: str, receiver: str, content: str, platform_time: int):
        topic = MessagingChannels.agent_dm(sender, receiver)
        entry = f"[Iter: {platform_time} | {current_time()}] {sender} → {receiver}: {content}"
        self.agent_dms.setdefault(topic, []).append(entry)

    def log_overseer_dm(self, sender: str, content: str, platform_time: int):
        topic = MessagingChannels.overseer_dm(sender)
        entry = f"[Iter: {platform_time} | {current_time()}] {sender} → Overseer: {content}"
        self.overseer_dms.setdefault(topic, []).append(entry)

    def get_gc_log(self) -> List[LLMMessage]:
        """Retrieve the general chat log."""
        return self.general_chat_log

    def get_agent_log(self, agent_id: str) -> List[LLMMessage]:
        """Retrieve logs for a specific agent."""
        return self.agent_logs.get(agent_id, [])

    def add_to_agent_log(self, agent_id: str, message: LLMMessage):
        """Utility to add messages directly into agent logs."""
        self.agent_logs.setdefault(agent_id, []).append(message)

    def add_to_general_log(self, message: LLMMessage):
        """Utility to add messages directly into general logs."""
        self.general_chat_log.append(message)
