from typing import Any, List
from autogen_core import FunctionCall
from autogen_core.models import LLMMessage
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, ConfigDict

from lasr_stealth_evals.library.logging import LogItem


class RequestToSpeak(BaseModel):
    pass


class Message(BaseModel):
    log: List[LogItem]


class FunctionCallRequest(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)  # type: ignore

    id: str
    call: FunctionCall
    tool: FunctionTool
    agent_id: str


class FunctionCallResponse(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)  # type: ignore

    result: Any
    approved: bool
    reason: str | None
    call: FunctionCall


class OverseerRequest(BaseModel):
    func_request: FunctionCallRequest


class OverseerResponse(BaseModel):
    func_response: FunctionCallResponse
