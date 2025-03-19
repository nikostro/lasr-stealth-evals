from typing import List, Any, Dict
from uuid import uuid4
from autogen_core import AgentId, FunctionCall, message_handler, MessageContext, RoutedAgent, DefaultTopicId
from autogen_core.models import LLMMessage, ChatCompletionClient, FunctionExecutionResult, UserMessage
from autogen_core.tools import FunctionTool

from src.lasr_stealth_evals.library.msg import (
    Message,
    RequestToSpeak,
    FunctionCallRequest,
    FunctionCallResponse,
    OverseerRequest,
    OverseerResponse,
)
from src.lasr_stealth_evals.library.logging import LogItem, Logger
from src.lasr_stealth_evals.example.env import Environment, AgentTools, OverseerTools

class Manager(RoutedAgent):
    def __init__(
            self,
            description: str,
            environment: Environment,
            agents: List["Agent"],
            logger: Logger,
            num_turns: int
        ):
        super().__init__(description)
        self._agents = agents
        self._agent_idx = 0
        self._env = environment
        self._logger = logger
        self._num_turns = num_turns
    
    @message_handler
    async def handle_message(self, msg: Message, ctx: MessageContext):
        """
        Handles messages from agents, logs them, and rotates to the next agent.
        
        Args:
            msg: Message containing logs from the agent
            ctx: Message context including the source agent
        """
        self._logger.extend(msg.log)
        self._agent_idx = (self._agent_idx + 1) % len(self._agents)
        await self.publish_message(
            RequestToSpeak(),
            topic_id=DefaultTopicId(str(self._agents[self._agent_idx].id), str(self.id))
        )

class Overseer(RoutedAgent):
    def __init__(
            self,
            description: str,
            manager_id: AgentId,
            tools: List[FunctionTool],
            logger: Logger
        ):
        super().__init__(description)
        self._manager_id = manager_id
        self._tools = tools
        self._logger = logger
        
    @message_handler
    async def handle_message(self, msg: OverseerRequest, ctx: MessageContext):
        """
        Evaluates function call requests and decides whether to approve them.
        
        Args:
            msg: The function call request to evaluate
            ctx: Message context
        """
        assert ctx.sender and ctx.sender == self._manager_id # This should only be called by the manager
        func_request = msg.func_request
        agent_logs = self._logger.get_agent_log(ctx.sender)
        is_approved = self._evaluate_function_call(func_request, agent_logs)
        reason = None if is_approved \
            else "Function call rejected by overseer"
        response = FunctionCallResponse(
            _id=func_request._id,
            result=None,  # Result will be filled by OverseerManager if approved
            approved=is_approved,
            reason=reason
        )
        await self.publish_message(
            OverseerResponse(func_response=response),
            topic_id=DefaultTopicId(str(self._manager_id), str(self.id))
        )
    
    def _evaluate_function_call(self, request: FunctionCallRequest, logs: List[LLMMessage]) -> bool:
        """
        Evaluates whether a function call should be allowed based on logs, rules and tools.
        
        Args:
            request: The function call request to evaluate
            logs: Historical logs for context
            
        Returns:
            Boolean indicating whether the function call is approved
        """
        # TODO: Implement evaluation logic
        return True


class OverseerManager(RoutedAgent):
    def __init__(self, description: str, overseer_topic_id: str, num_overseers: int):
        super().__init__(description)
        self._overseer_topic_id = overseer_topic_id
        self._num_overseers = num_overseers
        self._buffer: List[OverseerResponse] = []
        self._requests: Dict[str, FunctionCallRequest] = {}
        
    @message_handler
    async def handle_agent(self, msg: FunctionCallRequest, ctx: MessageContext):
        """
        Processes function call requests and forwards them to overseers if needed.
        
        Args:
            msg: The function call request
            ctx: Message context
        """
        assert ctx.sender # This should be called with a sender attached
        if self.needs_approval(msg):
            self._requests[msg._id] = msg
            self._buffer = []
            await self.publish_message(
                OverseerRequest(func_request=msg),
                topic_id=DefaultTopicId(self._overseer_topic_id, str(self.id))
            )
        else:
            result = self._execute_function_call(msg)    
            await self.publish_message(
                FunctionCallResponse(
                    _id=msg._id,
                    result=result,
                    approved=True,
                    reason=None
                ),
                topic_id=DefaultTopicId(str(ctx.sender), str(self.id))
            )
    
    @message_handler
    async def handle_overseer(self, msg: OverseerResponse, ctx: MessageContext):
        """
        Processes responses from overseers and executes function calls if approved.
        
        Args:
            msg: The overseer's response
            ctx: Message context
        """
        self._buffer.append(msg)
        if len(self._buffer) >= self._num_overseers:
            func_response = msg.func_response
            request_id = func_response._id
            
            if request_id in self._requests:
                request = self._requests.pop(request_id)
                is_approved = all(resp.func_response.approved for resp in self._buffer)
                
                if is_approved:
                    # Execute the function call
                    result = self._execute_function_call(request)
                    
                    # Return success result
                    await self.publish_message(
                        FunctionCallResponse(
                            _id=request_id,
                            result=result,
                            approved=True,
                            reason=None
                        ),
                        topic_id=DefaultTopicId(str(request.agent_id), str(self.id))
                    )
                else:
                    reasons = [
                        resp.func_response.reason 
                        for resp in self._buffer 
                        if not resp.func_response.approved and resp.func_response.reason
                    ]
                    await self.publish_message(
                        FunctionCallResponse(
                            _id=request_id,
                            result=None,
                            approved=False,
                            reason="; ".join(reasons) or "Function call rejected"
                        ),
                        topic_id=DefaultTopicId(str(request.agent_id), str(self.id))
                    )
            self.buffer = []
    
    def needs_approval(self, request: FunctionCallRequest) -> bool:
        """
        Determines whether a function call needs overseer approval.
        
        Args:
            request: The function call request
            
        Returns:
            Boolean indicating whether approval is needed
        """
        return True
    
    def _execute_function_call(self, request: FunctionCallRequest) -> Any:
        """
        Executes an approved function call.
        
        Args:
            request: The approved function call request
            
        Returns:
            The result of the function call
        """
        return None


class Agent(RoutedAgent):
    def __init__(
            self,
            description: str,
            group_manager_id: AgentId,
            overseer_manager_id: AgentId,
            tools: List[FunctionTool],
            logger: Logger
        ):
        super().__init__(description)
        self._group_manager_id = group_manager_id
        self._overseer_manager_id = overseer_manager_id
        self._tools = tools
        self._logger = logger
        self._function_call_responses = []
        
    @message_handler
    async def handle_group_manager(self, msg: RequestToSpeak, ctx: MessageContext):
        """
        Generates actions when it's this agent's turn to speak.
        
        Args:
            msg: Request to speak
            ctx: Message context
        """
        chat_history = self._logger.get_gc_log()
        call = self._invoke(chat_history)
        request = FunctionCallRequest(
            _id=f"{uuid4()}",
            call=call[0], # TODO: Handle multiple calls
            tool=self._get_tool_for_action(call[0]),
            agent_id=str(self.id)
        )
        await self.publish_message(
            request,
            topic_id=DefaultTopicId(str(self._overseer_manager_id), str(self.id))
        )
    
    @message_handler
    async def handle_overseer_manager(self, msg: FunctionCallResponse, ctx: MessageContext):
        """
        Processes responses to function call requests.
        
        Args:
            msg: The function call response
            ctx: Message context
        """
        
        # Create a message to signal end of turn
        logs = [FunctionExecutionResult(content=str(msg), name="dummy", call_id="dummy", is_error=False)]
        logs = [
            LogItem(
                source=self.id,
                content=UserMessage(content=log.content, source=str(self.id)),
                _obs=[self.id]
            ) for log in logs
        ]
        logs = Message(log=logs)
        await self.publish_message(
            logs,
            topic_id=DefaultTopicId(str(self._group_manager_id), str(self.id))
        )
    
    def _invoke(self, history: List[LLMMessage]) -> List[FunctionCall]:
        """
        Generates an action based on chat history.
        
        Args:
            history: Chat history logs
            
        Returns:
            A function call object
        """
        # TODO: Implement action generation logic using LLM
        return [FunctionCall("dummy", "{}", "dummy")]
    
    def _get_tool_for_action(self, action: FunctionCall) -> FunctionTool:
        """
        Finds the appropriate tool for a given action.
        
        Args:
            action: The action to find a tool for
            
        Returns:
            The matching function tool
        """
        # Find matching tool
        for tool in self._tools:
            if tool.name == action.name:
                return tool
        
        raise ValueError(f"No tool found for action {action.name}")