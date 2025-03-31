from typing import List, Any, Dict, Optional, Annotated
from uuid import uuid4
import json
import asyncio
import logging

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import (
    AgentId,
    CancellationToken,
    FunctionCall,
    message_handler,
    MessageContext,
    RoutedAgent,
    DefaultTopicId,
    AgentType,
    TopicId,
)
from autogen_core.models import (
    LLMMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    UserMessage,
    SystemMessage,
    AssistantMessage,
)
from autogen_core.tools import FunctionTool, Tool

from msg import (
    SimpleMessage,
    RequestToSpeak,
    FunctionCallRequest,
    FunctionCallResponse,
    OverseerRequest,
    OverseerResponse,
)
import logging
from logger import Logger
from env import Environment
from channels import MessagingChannels



sys_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)




class Manager(RoutedAgent):
    def __init__(
        self, description: str, environment: Environment, agent_topics: List[AgentType], logger: Logger, num_turns: int
    ):
        super().__init__(description)
        self._agent_topics = agent_topics
        self._agent_idx = 0
        self._env = environment
        self._logger = logger
        self._num_turns = num_turns

    @message_handler
    async def start(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        sys_logger.error(
            f"Manager received request from {ctx.sender}. Sending to {self._agent_topics[self._agent_idx]}"
        )
        await self.publish_message(
            RequestToSpeak(), topic_id=TopicId(type=self._agent_topics[self._agent_idx].type, source="default")
        )

    @message_handler
    async def handle_message(self, message: SimpleMessage, ctx: MessageContext) -> None:
        sys_logger.info(
            f"Manager received message from {ctx.sender}: {message.content}. "
            f"Currently at idx={self._agent_idx}, sending to {self._agent_topics[self._agent_idx]}"
        )
        self._logger.log_public(ctx.sender.type, message.content, self._env._platform.time)

        # Rotate the index and log explicitly
        self._agent_idx = (self._agent_idx + 1) % len(self._agent_topics)
        next_agent_type = self._agent_topics[self._agent_idx].type
        sys_logger.info(f"Next agent_idx={self._agent_idx}, next agent_type={next_agent_type}")

        await self.publish_message(
            RequestToSpeak(),
            topic_id=TopicId(type=next_agent_type, source="manager")
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
    ):
        super().__init__(description)
        self._manager_type = manager_type
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
        sys_logger.error("Overseer")
        assert ctx.sender  # This should only be called by the manager
        func_request = message.func_request
        agent_logs = self._logger.get_agent_log(ctx.sender)
        is_approved = self._monitor(func_request, agent_logs)
        reason = None if is_approved else "Function call rejected by overseer"
        response = FunctionCallResponse(
            id=func_request.id,
            result=None,  # Result will be filled by OverseerManager if approved
            approved=is_approved,
            reason=reason,
        )
        await self.publish_message(
            OverseerResponse(func_response=response),
            topic_id=TopicId(type=str(self._manager_type), source=str(self.id)),
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
    def __init__(self, description: str, overseer_topic_type: str, num_overseers: int):
        super().__init__(description)
        self._overseer_topic_type = overseer_topic_type
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
        sys_logger.error("In OverseerManager")
        assert ctx.sender  # This should be called with a sender attached
        if self.needs_approval(message):
            self._requests[message.id] = message
            self._buffer = []
            await self.publish_message(
                OverseerRequest(func_request=message),
                topic_id=TopicId(type=self._overseer_topic_type, source=str(self.id)),
            )
        else:
            result = await self._execute_function_call(message, ctx.cancellation_token)
            await self.publish_message(
                FunctionCallResponse(id=message.id, result=result, approved=True, reason=None),
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
        sys_logger.error("Overseer Manager received")
        self._buffer.append(message)
        if len(self._buffer) >= self._num_overseers:
            func_response = message.func_response
            request_id = func_response.id

            if request_id in self._requests:
                request = self._requests.pop(request_id)
                agent_type = AgentId.from_str(request.agent_id).type
                is_approved = all(resp.func_response.approved for resp in self._buffer)

                if is_approved:
                    # Execute the function call
                    result = await self._execute_function_call(request, ctx.cancellation_token)

                    # Return success result
                    await self.publish_message(
                        FunctionCallResponse(id=request_id, result=result, approved=True, reason=None),
                        topic_id=TopicId(type=str(agent_type), source=str(self.id)),
                    )
                else:
                    reasons = [
                        resp.func_response.reason
                        for resp in self._buffer
                        if not resp.func_response.approved and resp.func_response.reason
                    ]
                    await self.publish_message(
                        FunctionCallResponse(
                            id=request_id,
                            result=None,
                            approved=False,
                            reason="; ".join(reasons) or "Function call rejected",
                        ),
                        topic_id=TopicId(type=str(agent_type), source=str(self.id)),
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
        return False  # Temporary

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
        _use_tools: Optional[List[str]] = [],
    ):
        super().__init__(description)
        self._manager_type = manager_type
        self._overseer_manager_type = overseer_manager_type
        self._model_client = model_client
        self._logger = logger
        self._env = environment
        self._system_message = system_message
        self._function_call_responses = []
        self._tools = self._init_tools(_use_tools or [])

    def _init_tools(self, _use_tools: List[str] = []) -> List[FunctionTool]:
        if _use_tools:
            pass
        tools = [
            FunctionTool(
                self.buy,
                description="Buys `quantity` amount of `asset_name` from the market platform. "
                            "You must have sufficient balance to do this operation.",
            ),
            FunctionTool(self.balance, description="Returns the current balance of the agent."),
            FunctionTool(
                self.sell,
                "Sells `quantity` amount of `asset_name` from the market platform. "
                "You must have sufficient holdings of the asset to do this operation.",
            ),
            FunctionTool(self.prices, description="Returns the current prices of assets traded on the market."),
        ]
        sys_logger.error(tools[0].schema)
        return tools

    @message_handler
    @message_handler
    async def handle_group_manager(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        sys_logger.error(f"Agent received message from {ctx.sender}")
        chat_history = [
                           self._system_message,
                       ] + self._logger.get_gc_log()

        response = await self._invoke(chat_history, ctx.cancellation_token)

        # Check if the response contains a function call or regular text
        if isinstance(response[0], FunctionCall):
            action_name = response[0].name
            tool = self._get_tool_for_action(action_name)
            request = FunctionCallRequest(
                id=f"{uuid4()}",
                call=response[0],
                tool=tool,
                agent_id=str(self.id),
            )
            await self.publish_message(
                request,
                topic_id=DefaultTopicId(type=self._overseer_manager_type.type, source=str(self.id))
            )
        else:
            # Handle regular text response gracefully
            message_content = response[0]
            await self.publish




    @message_handler
    async def handle_overseer_manager(self, message: FunctionCallResponse, ctx: MessageContext) -> None:
        """
        Processes responses to function call requests.

        Args:
            message: The function call response
            ctx: Message context
        """

        # Create a message to signal end of turn
        logs = [SimpleMessage(sender=str(self.id), content=str(message))]
        await self.publish_message(logs[0], topic_id=DefaultTopicId(self._manager_type.type, str(self.id)))


    sys_logger = logging.getLogger(__name__)

    async def _invoke(self, history, cancellation_token):
        try:
            result = await asyncio.wait_for(
                self._model_client.create(
                    messages=history,
                    tools=self._tools,
                    cancellation_token=cancellation_token
                ),
                timeout=15  # seconds
            )
        except asyncio.TimeoutError:
            sys_logger.error("Timeout calling OpenAI API")
            raise RuntimeError("OpenAI API call timed out.")

        return result.content





    def _get_tool_for_action(self, action_name: str) -> FunctionTool:
        """
        Finds the appropriate tool for a given action.

        Args:
            action: The action to find a tool for

        Returns:
            The matching function tool
        """

        for tool in self._tools:
            if tool.name == action_name:
                return tool
        raise ValueError(f"No tool found for action '{action_name}'")

    ### Agent API ###
    async def buy(
            self,
            asset_name: Annotated[str, "Name of the asset you want to buy. This must be specified."],
            quantity: Annotated[int, "Number of stocks of the asset you want to buy."
                                     " Must be an integer greater than 0."],
    ) -> Dict[str, Any]:
        assert self._env
        return self._env.buy(
            self.id.type,
            {asset_name: quantity},
        )

    async def sell(
            self,
            asset_name: Annotated[str, "Name of the asset you want to sell. This must be specified."],
            quantity: Annotated[int, "Number of stocks of the asset you want to sell. Must be an integer."],
    ):
        assert self._env
        return self._env.sell(
            self.id.type,
            {asset_name: quantity},
        )


    async def balance(self):
        assert self._env
        return self._env.balance(self.id.type)

    async def prices(self):
        assert self._env
        return self._env.prices()