from typing import Any, Optional
from autogen_core import FunctionCall
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, ConfigDict

class RequestToSpeak(BaseModel):
    pass

class SimpleMessage(BaseModel):
    sender: str
    content: str
    timestamp: Optional[str] = None

class FunctionCallRequest(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    id: str
    call: FunctionCall
    tool: FunctionTool
    agent_id: str

class FunctionCallResponse(BaseModel):
    id: str
    result: Any
    approved: bool
    reason: Optional[str] = None

class OverseerRequest(BaseModel):
    func_request: FunctionCallRequest

class OverseerResponse(BaseModel):
    func_response: FunctionCallResponse
