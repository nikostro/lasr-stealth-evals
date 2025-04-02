from typing import Dict, List

from pydantic import BaseModel

from lasr_stealth_evals.library.msg import FunctionCallRequest, OverseerResponse


class ManagerState(BaseModel):  # TODO factor out as unchangeable
    agent_idx: int = 0
    current_turn: int = 0

    def reset(self) -> None:
        """Reset the state to initial values"""
        self.agent_idx = 0
        self.current_turn = 0


class AgentState(BaseModel):
    pass


class OverseerManagerState(BaseModel):
    requests: Dict[str, FunctionCallRequest] = {}
    buffer: List[OverseerResponse] = []

    def reset(self) -> None:
        """Reset the state to initial values"""
        self.requests = {}
        self.buffer = []


class OverseerState(BaseModel):
    pass
