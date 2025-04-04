from pydantic import BaseModel


class ManagerState(BaseModel):
    agent_idx: int = 0
    current_turn: int = 0

    def reset(self) -> None:
        """Reset the state to initial values"""
        self.agent_idx = 0
        self.current_turn = 0
