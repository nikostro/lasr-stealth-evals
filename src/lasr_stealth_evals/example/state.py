from dataclasses import dataclass
from typing import List

from lasr_stealth_evals.library.msg import FunctionCallRequest, OverseerResponse


@dataclass
class ManagerState:  # TODO factor out as unchangeable
    agent_idx: int = 0
    current_turn: int = 0

    def reset(self) -> None:
        """Reset the state to initial values"""
        self.agent_idx = 0
        self.current_turn = 0


@dataclass
class AgentState:
    pass


@dataclass
class OverseerManagerState:
    requests: List[FunctionCallRequest] = []
    buffer: List[OverseerResponse] = []

    def reset(self) -> None:
        """Reset the state to initial values"""
        self.requests = []
        self.buffer = []


@dataclass
class OverseerState:
    pass
