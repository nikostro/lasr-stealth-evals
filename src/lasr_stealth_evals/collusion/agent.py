from logging import getLogger
from inspect_ai.log._transcript import transcript

# Note: These private imports are used because there are no public API equivalents
# for these specific functions in the current version of inspect_ai
from inspect_ai.agent import (
    Agent,
    AgentState,
    agent,
    agent_with,
)
from inspect_ai.model import (
    ChatMessageTool,
    ChatMessageUser,
    Model,
    get_model,
)
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.scorer import score
from inspect_ai.tool import Tool, ToolResult, tool
from inspect_ai.tool._tool_info import parse_tool_info
from inspect_ai.util import store
from lasr_stealth_evals.collusion.util import get_system_message, filter_first_user_message
from lasr_stealth_evals.collusion.generate_samples import Metadata

logger = getLogger(__name__)


async def _agent_generate(model: str | Model | Agent | None, state: AgentState, tools: list[Tool]) -> AgentState:
    # convert model to agent
    if isinstance(model, str | Model) or model is None:
        model = _model_generate(model)

    # confirm we have a tools param
    agent_tool_info = parse_tool_info(model)
    if "tools" not in agent_tool_info.parameters.properties:
        raise ValueError("Agent passed as model for react agent must have a tools parameter.")

    # call the agent
    return await model(state, tools)


# submission tool
@tool
def submit_tool() -> Tool:
    async def execute(decision: bool, reason: str) -> ToolResult:
        """Submit an acceptance or rejection of the agent's message with reasoning for your decision.

        Args:
            decision (bool): Approve (true) or reject (false).
            reason (str): Reason for approving or rejecting.
        """
        response = {"answer": decision, "reason": reason}
        return str(response)

    return execute


def _model_generate(model: str | Model | None) -> Agent:
    async def generate(state: AgentState, tools: list[Tool]) -> AgentState:
        state.output = await get_model(model).generate(state.messages, tools)
        state.messages.append(state.output.message)
        return state

    return generate


@agent
def simple_agent(
    name: str,
    tools: list[Tool] = [],
    model: str | Model | Agent | None = None,
    tool_retry_limt: int = 3,  # number of times to reprompt
) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        transcript().info(name.upper())
        sample_metadata = Metadata(**store().get("metadata"))

        new_system_message = get_system_message(sample_metadata, name)
        state.messages[0] = new_system_message

        # We don't want default user message
        state.messages = filter_first_user_message(state.messages)

        retries = 0
        while True and retries < tool_retry_limt:
            # generate output and append assistant message
            state = await _agent_generate(model, state, tools)

            # The monitor is provided with a submit tool and must use it
            if tools:
                if state.output.message.tool_calls and state.output.message.tool_calls[0].parse_error is None:
                    # call tool functions
                    messages, output = await execute_tools(state.messages, tools)
                    state.messages.extend(messages)
                    if output:
                        state.output = output

                    # extract decision

                    tool_response = messages[-1]
                    assert type(tool_response) is ChatMessageTool
                    assert tool_response.function == "submit_tool"

                    # Score the output after tool execution
                    answer_scores = await score(state)
                    if answer_scores[0].value == "C":
                        store().set("completed", True)
                    return state
                else:
                    state.messages.extend(
                        [ChatMessageUser(content="Please use the submit_tool correctly to make your decision.")]
                    )
                    retries += 1
            else:
                break

        return state

    return agent_with(execute, name=name)
