from typing import List, Any

from autogen_core import FunctionCall
from autogen_core.models import LLMMessage
from autogen_core.tools import FunctionTool
from pydantic import BaseModel

class RequestToSpeak(BaseModel):
    pass

class Message(BaseModel):
    log: List[LLMMessage]

class FunctionCallRequest(BaseModel):
    _id: str
    call: FunctionCall
    tool: FunctionTool
    agent_id: str

class FunctionCallResponse(BaseModel):
    _id: str
    result: Any
    approved: bool
    reason: str | None

class OverseerRequest(BaseModel):
    func_request: FunctionCallRequest

class OverseerResponse(BaseModel):
    func_response: FunctionCallResponse
