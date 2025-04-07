from typing import List

from autogen_core import (
    AgentType,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
)

from lasr_stealth_evals.library.env import BaseEnvironment
from lasr_stealth_evals.library.logging import Logger
from lasr_stealth_evals.library.msg import (
    Message,
    RequestToSpeak,
)
from lasr_stealth_evals.library.state import ManagerState


class Manager(RoutedAgent):
    """A manager agent that coordinates communication between other agents in the system.

    This manager handles the turn-based communication flow between agents, managing their
    speaking order and tracking the overall conversation state.

    Args:
        description: A description of the manager's role
        environment: The environment the manager operates in
        agent_topics: List of agent types that the manager coordinates
        logger: Logger instance for tracking conversation
        num_turns: Number of conversation turns to run
        state: The manager's state tracking conversation progress
    """

    def __init__(
        self,
        description: str,
        environment: BaseEnvironment,
        agent_topics: List[AgentType],
        logger: Logger,
        num_turns: int,
        state: ManagerState,
        pretty_print: bool = True,
    ):
        super().__init__(description)
        self._agent_topics = agent_topics
        self._env = environment
        self._logger = logger
        self._num_turns = num_turns
        self._state = state
        self._pretty_print = pretty_print

    @message_handler
    async def start(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        """Initiates the conversation by requesting the first agent to speak.

        This method is called when the manager receives an initial RequestToSpeak message.
        It publishes a new RequestToSpeak message to the first agent in the sequence.

        Args:
            message: The initial RequestToSpeak message
            ctx: Message context containing sender information
        """
        await self.publish_message(
            RequestToSpeak(), topic_id=TopicId(type=self._agent_topics[self._state.agent_idx].type, source="default")
        )

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        """Processes messages from agents and manages the conversation flow.

        This method:
        1. Logs the received message
        2. Updates the agent index to move to the next agent
        3. Ticks the environment if all agents have spoken
        4. Requests the next agent to speak if the conversation should continue

        Args:
            message: The message received from an agent
            ctx: Message context containing sender information
        """
        assert ctx.sender
        self._logger.extend(message.log)
        self._state.agent_idx += 1
        if self._state.agent_idx >= len(self._agent_topics):
            self._state.current_turn += 1
            if self._pretty_print:
                self._logger.pretty_print()
            self._logger.tick(self._env.state())  # NOTE: tick for logging must happen before env tick
            self._env.tick()
        self._state.agent_idx %= len(self._agent_topics)
        if self._state.current_turn >= self._num_turns:
            self._logger.save_log()
            return
        await self.publish_message(
            RequestToSpeak(),
            topic_id=TopicId(type=str(self._agent_topics[self._state.agent_idx].type), source=str(self.id)),
        )
