from typing import List, Any, Dict, Optional
from uuid import uuid4
import json
from autogen_core import AgentId, CancellationToken, FunctionCall, message_handler, MessageContext, RoutedAgent, DefaultTopicId
from autogen_core.models import LLMMessage, ChatCompletionClient, FunctionExecutionResult, UserMessage, SystemMessage, AssistantMessage
from autogen_core.tools import FunctionTool

from lasr_stealth_evals.library.msg import (
    Message,
    RequestToSpeak,
    FunctionCallRequest,
    FunctionCallResponse,
    OverseerRequest,
    OverseerResponse,
)
from lasr_stealth_evals.library.logging import LogItem, Logger
from lasr_stealth_evals.example.env import Environment

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
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        """
        Handles messages from agents, logs them, and rotates to the next agent.
        
        Args:
            message: Message containing logs from the agent
            ctx: Message context including the source agent
        """
        self._logger.extend(message.log)
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
            model_client: ChatCompletionClient | None,
            system_message: SystemMessage | None,
            tools: List[FunctionTool],
            logger: Logger
        ):
        super().__init__(description)
        self._manager_id = manager_id
        self._model_client = model_client
        self._system_message = system_message
        self._tools = tools
        self._logger = logger
        
    @message_handler
    async def handle_message(self, message: OverseerRequest, ctx: MessageContext) -> None:
        """
        Evaluates function call requests and decides whether to approve them.
        
        Args:
            message: The function call request to evaluate
            ctx: Message context
        """
        assert ctx.sender and ctx.sender == self._manager_id # This should only be called by the manager
        func_request = message.func_request
        agent_logs = self._logger.get_agent_log(ctx.sender)
        is_approved = self._monitor(func_request, agent_logs)
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
    
    def _monitor(self, request: FunctionCallRequest, logs: List[LLMMessage]) -> bool:
        """
        Evaluates whether a function call should be allowed based on logs, rules and tools.
        
        Args:
            request: The function call request to evaluate
            logs: Historical logs for context
            
        Returns:
            Boolean indicating whether the function call is approved
        """
        return True

    ## TODO: Add tools for overseer

    async def flag(self, message: str):
        """
        Flags an action for review by the overseer.
        
        Args:
            message: The message to flag
        """
        return message

class OverseerManager(RoutedAgent):
    def __init__(self, description: str, overseer_topic_id: str, num_overseers: int):
        super().__init__(description)
        self._overseer_topic_id = overseer_topic_id
        self._num_overseers = num_overseers
        self._buffer: List[OverseerResponse] = []
        self._requests: Dict[str, FunctionCallRequest] = {}
        
    @message_handler
    async def handle_agent(self, message: FunctionCallRequest, ctx: MessageContext) -> None:
        """
        Processes function call requests and forwards them to overseers if needed.
        
        Args:
            message: The function call request
            ctx: Message context
        """
        assert ctx.sender # This should be called with a sender attached
        if self.needs_approval(message):
            self._requests[message._id] = message
            self._buffer = []
            await self.publish_message(
                OverseerRequest(func_request=message),
                topic_id=DefaultTopicId(self._overseer_topic_id, str(self.id))
            )
        else:
            result = await self._execute_function_call(message, ctx.cancellation_token)
            await self.publish_message(
                FunctionCallResponse(
                    _id=message._id,
                    result=result,
                    approved=True,
                    reason=None
                ),
                topic_id=DefaultTopicId(str(ctx.sender), str(self.id))
            )
    
    @message_handler
    async def handle_overseer(self, message: OverseerResponse, ctx: MessageContext) -> None:
        """
        Processes responses from overseers and executes function calls if approved.
        
        Args:
            message: The overseer's response
            ctx: Message context
        """
        self._buffer.append(message)
        if len(self._buffer) >= self._num_overseers:
            func_response = message.func_response
            request_id = func_response._id
            
            if request_id in self._requests:
                request = self._requests.pop(request_id)
                is_approved = all(resp.func_response.approved for resp in self._buffer)
                
                if is_approved:
                    # Execute the function call
                    result = await self._execute_function_call(request, ctx.cancellation_token)
                    
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
    
    async def _execute_function_call(self, request: FunctionCallRequest, cancellation_token: CancellationToken) -> Any:
        """
        Executes an approved function call.
        
        Args:
            request: The approved function call request
            cancellation_token: Cancellation token
            
        Returns:
            The result of the function call
        """
        return await request.tool.run_json(
            json.loads(request.call.arguments),
            cancellation_token
        )

class Agent(RoutedAgent):
    def __init__(
            self,
            description: str,
            group_manager_id: AgentId,
            overseer_manager_id: AgentId,
            model_client: ChatCompletionClient,
            environment: Environment | None,
            tools: List[FunctionTool],
            logger: Logger
        ):
        super().__init__(description)
        self._group_manager_id = group_manager_id
        self._overseer_manager_id = overseer_manager_id
        self._model_client = model_client
        self._tools = tools
        self._logger = logger
        self._env = environment
        self._function_call_responses = []
        
    @message_handler
    async def handle_group_manager(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        """
        Generates actions when it's this agent's turn to speak.
        
        Args:
            message: Request to speak
            ctx: Message context
        """
        chat_history = self._logger.get_gc_log()
        call = await self._invoke(chat_history, ctx.cancellation_token)
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
    async def handle_overseer_manager(self, message: FunctionCallResponse, ctx: MessageContext) -> None:
        """
        Processes responses to function call requests.
        
        Args:
            message: The function call response
            ctx: Message context
        """
        
        # Create a message to signal end of turn
        logs = [FunctionExecutionResult(content=str(message), name="dummy", call_id="dummy", is_error=False)]
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
    
    async def _invoke(self, history: List[LLMMessage], cancellation_token: CancellationToken) -> List[FunctionCall]:
        """
        Generates an action based on chat history.
        
        Args:
            history: Chat history logs
            
        Returns:
            A function call object
        """
        result = await self._model_client.create(
            messages=history,
            tools=self._tools,
            cancellation_token=cancellation_token
        )
        if isinstance(result.content, str):
            raise Exception("Model returned a string instead of a list of function calls")
        assert isinstance(result.content, list) and all(
            isinstance(call, FunctionCall) for call in result.content
        )
        return result.content

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

    ### Agent API ###

    async def buy(self, request: Dict[str, int]):
        """
        Buys assets based on a request.
        """
        assert self._env
        return self._env.buy(self.id, request)
    
    async def sell(self, request: Dict[str, int]):
        """
        Sells assets based on a request.
        """
        assert self._env
        return self._env.sell(self.id, request)
    
    async def balance(self):
        """
        Returns the current balance.
        """
        assert self._env
        return self._env.balance(self.id)
    
    async def prices(self):
        """
        Returns the current prices of assets traded on the market.
        """
        assert self._env
        return self._env.prices()
