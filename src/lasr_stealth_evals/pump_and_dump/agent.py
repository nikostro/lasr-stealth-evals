from typing import List, Any, Dict, Optional, Annotated
from uuid import uuid4
import json
import asyncio
from channels import MessagingChannels
from dataclasses import dataclass



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



from typing import List, Any, Dict, Optional, Annotated, ClassVar
from uuid import uuid4
import json
import asyncio
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
    UserMessage,
    SystemMessage,
    AssistantMessage,
)
from autogen_core.tools import FunctionTool
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

sys_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class ManagerState:
    agent_idx: int = 0
    current_turn: int = 0
    _instance: ClassVar["ManagerState | None"] = None

    @classmethod
    def get_instance(cls) -> "ManagerState":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        self.agent_idx = 0
        self.current_turn = 0

class Manager(RoutedAgent):
    def __init__(self, description: str, environment: Environment, agent_topics: List[AgentType], logger: Logger, num_turns: int):
        super().__init__(description)
        self._agent_topics = agent_topics
        self._env = environment
        self._logger = logger
        self._num_turns = num_turns
        self._state = ManagerState.get_instance()

    @message_handler
    async def start(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        sys_logger.info("Simulation started.")
        first_agent_type = self._agent_topics[self._state.agent_idx].type
        await self.publish_message(RequestToSpeak(), topic_id=TopicId(type=first_agent_type, source="manager"))

    @message_handler
    async def handle_message(self, message: SimpleMessage, ctx: MessageContext) -> None:
        sys_logger.info(f"Received message from {ctx.sender}: {message.content}")
        self._logger.log_public(ctx.sender.type, message.content, self._env._platform.time)

        current_agent_type = self._agent_topics[self._state.agent_idx].type
        trader_id = AgentId(current_agent_type, "manager")
        trader = self._env._traders[trader_id]
        trader.actions_left -= 1

        if trader.actions_left <= 0:
            self._state.agent_idx += 1
            if self._state.agent_idx >= len(self._agent_topics):
                misaligned_agents = [agent_id for agent_id, trader in self._env._traders.items() if not trader.aligned]
                self._env.tick(self._env._platform.structured_trade_log, misaligned_agents)
                for trader in self._env._traders.values():
                    trader.actions_left = 5
                self._state.agent_idx = 0
                self._state.current_turn += 1
                if self._state.current_turn >= self._num_turns:
                    sys_logger.info("Max turns reached. Ending simulation.")
                    return

        next_agent_type = self._agent_topics[self._state.agent_idx].type
        await self.publish_message(RequestToSpeak(), topic_id=TopicId(type=next_agent_type, source="manager"))

class Overseer(RoutedAgent):
    def __init__(self, description: str, manager_type: AgentType, model_client: ChatCompletionClient | None,
                 system_message: SystemMessage | None, tools: List[FunctionTool], logger: Logger):
        super().__init__(description)
        self._manager_type = manager_type
        self._model_client = model_client
        self._system_message = system_message
        self._tools = tools
        self._logger = logger

    @message_handler
    async def handle_message(self, message: OverseerRequest, ctx: MessageContext) -> None:
        assert ctx.sender
        sys_logger.info("Overseer handling request.")
        func_request = message.func_request
        approved = self._monitor(func_request, self._logger.get_agent_log(ctx.sender))
        reason = None if approved else "Rejected by Overseer"
        await self.publish_message(
            OverseerResponse(func_response=FunctionCallResponse(id=func_request.id, approved=approved, reason=reason)),
            topic_id=TopicId(type=str(self._manager_type), source=str(self.id)),
        )

    def _monitor(self, request: FunctionCallRequest, logs: List[LLMMessage]) -> bool:
        return True

class OverseerManager(RoutedAgent):
    def __init__(self, description: str, overseer_topic_type: str, num_overseers: int):
        super().__init__(description)
        self._overseer_topic_type = overseer_topic_type
        self._num_overseers = num_overseers
        self._buffer = []
        self._requests = {}

    @message_handler
    async def handle_agent(self, message: FunctionCallRequest, ctx: MessageContext) -> None:
        if not self.needs_approval(message):
            result = await message.tool.run_json(json.loads(message.call.arguments), ctx.cancellation_token)
            await self.publish_message(FunctionCallResponse(id=message.id, result=result, approved=True, reason=None), topic_id=TopicId(type=str(ctx.sender.type), source=str(self.id)))

    def needs_approval(self, request: FunctionCallRequest) -> bool:
        return False
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
            _use_tools=None,
    ):
        super().__init__(description)
        if _use_tools is None:
            _use_tools = []
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
            await self.publish_message(
                SimpleMessage(sender=str(self.id), content=message_content),
                topic_id=DefaultTopicId(type=self._manager_type.type, source=str(self.id))
            )





    @message_handler
    async def handle_overseer_manager(self, message: FunctionCallResponse, ctx: MessageContext) -> None:
        """
        Processes responses to function call requests.

        Args:
            message: The function call response
            ctx: Message context
        """
        if message.approved:
            # Execute the function call and get the result
            result = await message.tool.run_json(json.loads(message.call.arguments), ctx.cancellation_token)
            # Send the result back to the manager
            await self.publish_message(
                SimpleMessage(sender=str(self.id), content=f"Function executed successfully: {result}"),
                topic_id=DefaultTopicId(type=self._manager_type.type, source=str(self.id))
            )
        else:
            # Send rejection message back to the manager
            await self.publish_message(
                SimpleMessage(sender=str(self.id), content=f"Function call rejected: {message.reason}"),
                topic_id=DefaultTopicId(type=self._manager_type.type, source=str(self.id))
            )


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
        return self._env.balance(self.id)

    async def prices(self):
        assert self._env
        raw_prices = self._env.prices()
        # Convert np.float64 to Python floats explicitly
        return {key: float(value) for key, value in raw_prices.items()}
