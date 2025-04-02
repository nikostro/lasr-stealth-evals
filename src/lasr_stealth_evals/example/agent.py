import json
from inspect import getdoc
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple

from autogen_core import (
    AgentId,
    AgentType,
    CancellationToken,
    DefaultTopicId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
)
from autogen_core.tools import FunctionTool

from lasr_stealth_evals.example.env import Environment
from lasr_stealth_evals.example.state import AgentState, ManagerState, OverseerManagerState, OverseerState
from lasr_stealth_evals.library.logging import GROUP_CHAT, Logger, LogItem
from lasr_stealth_evals.library.msg import (
    FunctionCallRequest,
    FunctionCallResponse,
    Message,
    OverseerRequest,
    OverseerResponse,
    RequestToSpeak,
)
from lasr_stealth_evals.library.system_logger import get_logger

sys_logger = get_logger(__name__)


def format_gc_log(messages: List[LLMMessage]) -> AssistantMessage:
    assert all(isinstance(message, AssistantMessage) for message in messages)
    return AssistantMessage(
        content="\n\n".join(f"**{message.source or 'Unknown'}**: {message.content}" for message in messages),  # type: ignore
        source="group_chat",
    )


def join_messages(messages: Sequence[LLMMessage], source: str) -> LLMMessage:
    return AssistantMessage(
        content="\n".join(
            str(message.content) if isinstance(message.content, (str, list)) else repr(message.content)
            for message in messages
        ),
        source=source,
    )


class Manager(RoutedAgent):  # TODO factor out as unchangeable
    def __init__(
        self,
        description: str,
        environment: Environment,
        agent_topics: List[AgentType],
        logger: Logger,
        num_turns: int,
        state: ManagerState,
    ):
        super().__init__(description)
        self._agent_topics = agent_topics
        self._env = environment
        self._logger = logger
        self._num_turns = num_turns
        self._state = state

    @message_handler
    async def start(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        await self.publish_message(
            RequestToSpeak(), topic_id=TopicId(type=self._agent_topics[self._state.agent_idx].type, source="default")
        )

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        assert ctx.sender
        self._logger.extend(message.log)
        self._logger.pretty_print()
        self._logger.save_log()
        self._state.agent_idx += 1
        if self._state.agent_idx >= len(self._agent_topics):
            self._state.current_turn += 1
            self._env.tick()
        self._state.agent_idx %= len(self._agent_topics)
        if self._state.current_turn >= self._num_turns:
            return
        await self.publish_message(
            RequestToSpeak(),
            topic_id=TopicId(type=str(self._agent_topics[self._state.agent_idx].type), source=str(self.id)),
        )


class Overseer(RoutedAgent):
    def __init__(
        self,
        description: str,
        manager_type: AgentType,
        model_client: ChatCompletionClient | None,
        system_message: SystemMessage | None,
        tools: List[FunctionTool],
        logger: Logger,
        state: OverseerState,
    ):
        super().__init__(description)
        self._manager_type = manager_type
        self._model_client = model_client
        self._system_message = system_message
        self._use_tools = tools
        self._logger = logger
        self._state = state

    @message_handler
    async def handle_message(self, message: OverseerRequest, ctx: MessageContext) -> None:
        """
        Evaluates function call requests and decides whether to approve them.

        Args:
            message: The function call request to evaluate
            ctx: Message context
        """
        assert ctx.sender
        func_request = message.func_request
        agent_logs = self._logger.get_agent_log(ctx.sender.type)
        is_approved = self._monitor(func_request, agent_logs)
        reason = None if is_approved else "Function call rejected by overseer"
        response = FunctionCallResponse(
            result=None,  # Result will be filled by OverseerManager if approved
            approved=is_approved,
            call=func_request.call,
            reason=reason,
        )
        await self.publish_message(
            OverseerResponse(func_response=response),
            topic_id=TopicId(type=self._manager_type.type, source=str(self.id)),
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
    def __init__(self, description: str, overseer_topic_type: str, num_overseers: int, state: OverseerManagerState):
        super().__init__(description)
        self._overseer_topic_type = overseer_topic_type
        self._num_overseers = num_overseers
        self._state = state

    @message_handler
    async def handle_agent(self, message: FunctionCallRequest, ctx: MessageContext) -> None:
        """
        Processes function call requests and forwards them to overseers if needed.

        Args:
            message: The function call request
            ctx: Message context
        """
        assert ctx.sender
        if self.needs_approval(message):
            self._state.requests[message.call.id] = message
            self._state.buffer = []
            await self.publish_message(
                OverseerRequest(func_request=message),
                topic_id=TopicId(type=self._overseer_topic_type, source=str(self.id)),
            )
        else:
            result = await self._execute_function_call(message, ctx.cancellation_token)
            await self.publish_message(
                FunctionCallResponse(call=message.call, result=result, approved=True, reason=None),
                topic_id=TopicId(type=str(ctx.sender.type), source=str(self.id)),
            )

    @message_handler
    async def handle_overseer(self, message: OverseerResponse, ctx: MessageContext) -> None:
        """
        Processes responses from overseers and executes function calls if approved.

        Args:
            message: The overseer's response
            ctx: Message context
        """
        self._state.buffer.append(message)
        if len(self._state.buffer) >= self._num_overseers:
            func_response = message.func_response
            request_id = func_response.call.id

            if request_id in self._state.requests:
                request = self._state.requests.pop(request_id)
                agent_type = AgentId.from_str(request.agent_type).type
                is_approved = all(resp.func_response.approved for resp in self._state.buffer)

                if is_approved:
                    # Execute the function call
                    result = await self._execute_function_call(request, ctx.cancellation_token)

                    # Return success result
                    await self.publish_message(
                        FunctionCallResponse(call=request.call, result=result, approved=True, reason=None),
                        topic_id=TopicId(type=str(agent_type), source=str(self.id)),
                    )
                else:
                    reasons = [
                        resp.func_response.reason
                        for resp in self._state.buffer
                        if not resp.func_response.approved and resp.func_response.reason
                    ]
                    await self.publish_message(
                        FunctionCallResponse(
                            call=request.call,
                            result=None,
                            approved=False,
                            reason="; ".join(reasons) or "Function call rejected",
                        ),
                        topic_id=TopicId(type=str(agent_type), source=str(self.id)),
                    )
            self._state.buffer = []

    def needs_approval(self, request: FunctionCallRequest) -> bool:
        """
        Determines whether a function call needs overseer approval.

        Args:
            request: The function call request

        Returns:
            Boolean indicating whether approval is needed
        """
        return True  # Temporary

    async def _execute_function_call(self, request: FunctionCallRequest, cancellation_token: CancellationToken) -> Any:
        """
        Executes an approved function call.

        Args:
            request: The approved function call request
            cancellation_token: Cancellation token

        Returns:
            The result of the function call
        """
        return await request.tool.run_json(json.loads(request.call.arguments), cancellation_token)


def agent_tool(f: Any) -> Any:
    """Mark functions in the agent as tools"""
    f._is_agent_tool = True
    return f


class Agent(RoutedAgent):
    def __init__(
        self,
        description: str,
        manager_type: AgentType,
        overseer_manager_type: AgentType,
        model_client: ChatCompletionClient,
        environment: Environment | None,
        system_message: SystemMessage,
        logger: Logger,
        state: AgentState,
        _use_tools: List[str] = [],
        _allow_thought: bool = True,
    ):
        super().__init__(description)
        self._manager_type = manager_type
        self._overseer_manager_type = overseer_manager_type
        self._model_client = model_client
        self._logger = logger
        self._env = environment
        self._system_message = system_message
        self._tools = self._init_tools(_use_tools)
        self._allow_thought = _allow_thought
        self._state = state

    def _init_tools(self, _tools: list[str]) -> List[FunctionTool]:
        available_tools = self.get_agent_tool_names()

        # If no specific tools provided, use all available tools
        tools_to_add = _tools if _tools else available_tools

        # Verify specified tools exist (if any were specified)
        if _tools:
            assert all(
                tool in available_tools for tool in tools_to_add
            ), f"Some tools in {tools_to_add} are not in {available_tools}"

        # Create FunctionTools for each tool name
        tools = []
        for tool_name in tools_to_add:
            tool_method = getattr(self, tool_name)
            if hasattr(tool_method, "_is_agent_tool"):
                description = getdoc(tool_method) or f"Execute the {tool_name} function"
                tools.append(FunctionTool(tool_method, description=description))
        return tools

    @message_handler
    async def handle_group_manager(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        """
        Generates actions when it's this agent's turn to speak.

        Args:
            message: Request to speak
            ctx: Message context
        """
        assert ctx.sender
        thought, call = await self._invoke(ctx.cancellation_token)
        request = FunctionCallRequest(
            call=call,
            tool=self._get_tool_for_action(call),
            agent_type=str(self.id.type),
            thought=thought.content if thought else None,  # type: ignore
        )
        await self.publish_message(
            request, topic_id=DefaultTopicId(str(self._overseer_manager_type.type), str(self.id))
        )

    @message_handler
    async def handle_overseer_manager(self, message: FunctionCallResponse, ctx: MessageContext) -> None:
        """
        Processes responses to function call requests.

        Args:
            message: The function call response
            ctx: Message context
        """
        assert ctx.sender
        # Create a message to signal end of turn
        is_error = message.approved is False
        content = message.reason or "Function call rejected" if is_error else str(message.result)
        result = FunctionExecutionResult(
            content=content, name=message.call.name, call_id=message.call.id, is_error=is_error
        )
        logs = [
            LogItem(
                source=self.id.type,
                content=FunctionExecutionResultMessage(content=[result]),
                observability=[self.id.type],
            )
        ]
        logs = Message(log=logs)
        await self.publish_message(logs, topic_id=TopicId(self._manager_type.type, str(self.id)))

    async def _get_thought(self, cancellation_token: CancellationToken) -> AssistantMessage:
        history = [self._system_message] + [format_gc_log(self._logger.get_gc_log())]
        history += self._logger.get_agent_log(self.id.type)

        thought = [AssistantMessage(content="<thinking>", source=str(self.id.type))]
        result = await self._model_client.create(messages=history + thought, cancellation_token=cancellation_token)
        assert isinstance(result.content, str)
        thought += [AssistantMessage(content=result.content, source=str(self.id.type))]
        thought += [AssistantMessage(content="</thinking>", source=str(self.id.type))]
        thought = join_messages(thought, str(self.id.type))

        return thought  # type: ignore

    async def _get_action(self, cancellation_token: CancellationToken) -> Tuple[FunctionCall, AssistantMessage]:
        """
        Generates an action based on chat history.
        """
        history = [self._system_message] + [format_gc_log(self._logger.get_gc_log())]
        history += self._logger.get_agent_log(self.id.type)

        result = await self._model_client.create(
            messages=history + [AssistantMessage(content="<action>", source=str(self.id.type))],
            tools=self._tools,
            cancellation_token=cancellation_token,
        )
        if isinstance(result.content, str):
            sys_logger.warning(f"Expected list of function calls, got string: {result.content}")
        assert isinstance(result.content, list) and all(isinstance(call, FunctionCall) for call in result.content)
        if len(result.content) != 1:
            sys_logger.warning(f"Expected 1 action, got {len(result.content)}")
        action = [AssistantMessage(content=[result.content[0]], source="tools_calls")]

        return result.content[0], action[0]

    async def _invoke(self, cancellation_token: CancellationToken) -> Tuple[Optional[AssistantMessage], FunctionCall]:
        """
        Generates an action based on chat history.

        Args:
            history: Chat history logs

        Returns:
            A function call object
        """
        thought = None
        if self._allow_thought:
            thought = await self._get_thought(cancellation_token)
            self._logger.append(LogItem(source=self.id.type, content=thought, observability=[self.id.type]))
        action, action_message = await self._get_action(cancellation_token)
        self._logger.append(LogItem(source=self.id.type, content=action_message, observability=[self.id.type]))
        return (thought, action)

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
        # TODO add method to give feedback for bad tool calls
        raise ValueError(f"No tool found for action {action.name}")

    ### Agent API ###
    def get_agent_tool_names(self) -> List[str]:
        return [name for name, func in self.__class__.__dict__.items() if hasattr(func, "_is_agent_tool")]

    @agent_tool
    async def buy(
        self,
        asset_name: Annotated[str, "Name of the asset you want to buy. This must be specified."],
        quantity: Annotated[
            int, "Number of stocks of the asset you want to buy. This must be a integer greater than 0."
        ],
    ) -> Dict[str, Any]:
        """Buys `quantity` amount of `asset_name` from the market platform. You must have sufficient balance to do this operation."""
        assert self._env
        return self._env.buy(
            self.id.type,
            asset_name=asset_name,
            quantity=quantity,
        )

    @agent_tool
    async def sell(
        self,
        asset_name: Annotated[str, "Name of the asset you want to sell. This must be specified."],
        quantity: Annotated[int, "Number of stocks of the asset you want to sell. This must be an integer."],
    ):
        """Sells `quantity` amount of `asset_name` from the market platform. You must have sufficient holdings of the asset to do this operation."""
        assert self._env
        return self._env.sell(
            self.id.type,
            asset_name=asset_name,
            quantity=quantity,
        )

    @agent_tool
    async def balance(self):
        """Returns the current balance of the agent."""
        assert self._env
        return self._env.balance(self.id.type)

    @agent_tool
    async def prices(self):
        """Returns the current prices of assets traded on the market."""
        assert self._env
        return self._env.prices()

    @agent_tool
    async def send_message(self, content: str):
        """Sends a message to the group chat visible by all agents."""
        self._logger.append(
            LogItem(
                source=self.id.type,
                content=AssistantMessage(
                    content=content,
                    source=str(self.id.type),
                ),
                observability=[GROUP_CHAT],
            )
        )
