from logging import getLogger
from inspect_ai.log._transcript import transcript

# Note: These private imports are used because there are no public API equivalents
# for these specific functions in the current version of inspect_ai
from inspect_ai._util._async import is_callable_coroutine
from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentContinue,
    AgentPrompt,
    AgentState,
    AgentSubmit,
    agent,
    agent_with,
)
from inspect_ai.agent._handoff import has_handoff
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Model,
    get_model,
)
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.scorer import score
from inspect_ai.tool import Tool, ToolCall, ToolResult, tool
from inspect_ai.tool._tool_info import parse_tool_info
from inspect_ai.util import store
from lasr_stealth_evals.collusion.util import update_system_message, filter_first_user_message
from lasr_stealth_evals.collusion.generate_samples import Config

logger = getLogger(__name__)


@agent
def react(
    *,
    name: str | None = None,
    description: str | None = None,
    prompt: str | AgentPrompt | None = AgentPrompt(),
    tools: list[Tool] | None = None,
    model: str | Model | Agent | None = None,
    attempts: int | AgentAttempts = 1,
    submit: AgentSubmit = AgentSubmit(),
    on_continue: str | AgentContinue | None = None,
) -> Agent:
    """Extensible ReAct agent based on the paper [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).

    Provide a `name` and `description` for the agent if you plan on using it
    in a multi-agent system (this is so other agents can clearly identify
    its name and purpose). These fields are not required when using `react()`
    as a top-level solver.

    The agent runs a tool use loop until the model submits an answer using the
    `submit()` tool. Use `instructions` to tailor the agent's system message
    (the default `instructions` provides a basic ReAct prompt).

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    By default, the model will be urged to continue if it fails to call
    a tool. Customise this behavior using the `on_continue` option.

    Args:
       name: Agent name (required when using with `handoff()` or `as_tool()`)
       description: Agent description (required when using with `handoff()` or `as_tool()`)
       prompt: Prompt for agent. Includes agent-specific contextual `instructions`
          as well as an optional `assistant_prompt` and `handoff_prompt` (for agents
          that use handoffs). both are provided by default but can be removed or
          customized). Pass `str` to specify the instructions and use the defaults
          for handoff and prompt messages.
       tools: Tools available for the agent.
       model: Model to use for agent (defaults to currently evaluated model).
       attempts: Configure agent to make multiple attempts.
       submit: Configure submit tool used by agent.
       on_continue: Message to play back to the model to urge it to continue.
          Optionally, can also be an async function to call to determine whether
          the loop should continue (executed on every turn) and what message
          to play back.

    Returns:
        ReAct agent.
    """
    # resolve prompt / system message
    prompt = AgentPrompt(prompt, handoff_prompt="", assistant_prompt="") if isinstance(prompt, str) else prompt
    if prompt:
        prompt_lines: list[str] = []
        if prompt.instructions:
            prompt_lines.append(prompt.instructions)
        if prompt.handoff_prompt and has_handoff(tools):
            prompt_lines.append(prompt.handoff_prompt)
        if prompt.assistant_prompt:
            prompt_lines.append(prompt.assistant_prompt)
        prompt_content = "\n\n".join(prompt_lines).format(submit=submit.name)
        system_message: ChatMessage | None = ChatMessageSystem(content=prompt_content)
    else:
        system_message = None

    # resolve on_continue
    if on_continue is None:
        on_continue = "If you believe you have completed the task, please call the `submit()` tool with your answer."
    if isinstance(on_continue, str):
        no_tools_continue_message = on_continue

        async def no_tools_continue(state: AgentState) -> bool | str:
            if state.output is None or not state.output.message.tool_calls:
                return no_tools_continue_message
            else:
                return True

        on_continue = no_tools_continue

    # validate that on_continue is async
    if not is_callable_coroutine(on_continue):
        raise ValueError("The on_continue function must be async.")

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    # helper to see if there is a submit tool call
    def submitted_answer(tool_calls: list[ToolCall] | None) -> str | None:
        for tool_call in tool_calls or []:
            if tool_call.function == submit.name and tool_call.parse_error is None:
                return str(tool_call.arguments["answer"])
        return None

    # resolve tools
    tools = tools or []

    async def execute(state: AgentState) -> AgentState:
        """
        agent_name (str): Name of the agent to retrieve its system message.
        """
        # prepend system message if we have one
        sample_metadata = Config(**store().get("metadata"))

        if system_message:
            new_system_message = update_system_message(sample_metadata, system_message)
            state.messages[0] = new_system_message

        state.messages = filter_first_user_message(state.messages)

        # track attempts
        attempt_count = 0

        # main loop = will terminate after submit (subject to max_attempts)
        # or if a message or token limit is hit
        while True:
            # generate output and append assistant message
            state = await _agent_generate(model, state, tools)

            # check for context window overflow
            if state.output.stop_reason == "model_length":
                transcript().info("Agent terminated: model context window exceeded")
                break

            # check for a submission
            answer = submitted_answer(state.output.message.tool_calls)
            if answer is not None:
                # remove the tool call and set the output to the answer for scoring
                state.output.message.tool_calls = None
                state.output.completion = f"{state.output.completion}\n\n{answer}".strip()

                # exit if we are at max_attempts
                attempt_count += 1
                if attempt_count >= attempts.attempts:
                    break

                # exit if the submission is successful
                answer_scores = await score(state)
                if attempts.score_value(answer_scores[0].value) == 1.0:
                    break

                # otherwise notify the model that it was incorrect and continue
                else:
                    if callable(attempts.incorrect_message):
                        if not is_callable_coroutine(attempts.incorrect_message):
                            raise ValueError("The incorrect_message function must be async.")
                        response_message: str = await attempts.incorrect_message(state, answer_scores)
                    else:
                        response_message = attempts.incorrect_message

                    state.messages.append(ChatMessageUser(content=response_message))

            # no submitted answer, call tools and evaluate whether we should continue
            else:
                if state.output.message.tool_calls:
                    # call tool functions
                    messages, output = await execute_tools(state.messages, tools)
                    state.messages.extend(messages)
                    if output:
                        state.output = output

                # check if we should continue....
                do_continue = await on_continue(state)
                if isinstance(do_continue, str):
                    state.messages.append(ChatMessageUser(content=do_continue))
                elif do_continue is False:
                    break

        return state

    if name is not None or description is not None:
        return agent_with(execute, name=name, description=description)
    else:
        return execute


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
    system_message: ChatMessage | None = ChatMessageSystem(content=name)

    async def execute(state: AgentState) -> AgentState:
        transcript().info(name.upper())
        sample_metadata = Config(**store().get("metadata"))

        new_system_message = update_system_message(sample_metadata, system_message)
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
