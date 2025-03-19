from typing import List

from autogen_core import AgentId
from autogen_core.models import LLMMessage
from pydantic import BaseModel

GROUP_CHAT = "GROUP_CHAT"

class LogItem(BaseModel):
	source: AgentId
	content: LLMMessage
	_obs: List[AgentId]

class Logger:
	def __init__(self):
		self.log: List[LogItem] = []
	
	def get_agent_log(self, agent_id: AgentId) -> List[LLMMessage]:
		return [
			item.content for item in self.log
			if agent_id in item._obs
		]
	 
	def get_gc_log(self) -> List[LLMMessage]:
		return [
			item.content for item in self.log
			if GROUP_CHAT in item._obs
		]
	
	def append(self, log: LogItem):
		self.extend([log])

	def extend(self, log: List[LogItem]):
		self.log.extend(log)

