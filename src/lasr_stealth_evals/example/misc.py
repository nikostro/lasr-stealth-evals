import asyncio
import json
from dataclasses import dataclass
from typing import List

from autogen_core import (
    AgentId,
    CancellationToken,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from typing import Annotated


async def buy(
    request: Annotated[
        dict[str, int],
        "Provide a buy request in the form of a dict, with asset name str as key, and int quantity as value.",
    ],
) -> dict[str, int]:
    """Return stock price of named stock"""
    # Returns a random stock price for demonstration purposes.
    return request


async def main():
    # Create the OpenAI chat completion client. Using OPENAI_API_KEY from environment variable.
    client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    # Create a function tool.
    stock_price_tool = FunctionTool(buy, description="Buy the given quantity of stocks")
    print(stock_price_tool.schema)

    # Run the tool.
    cancellation_token = CancellationToken()

    # Create a user message.
    user_message = UserMessage(content="Buy 112 units of AAPL.", source="user")

    # Run the chat completion with the stock_price_tool defined above.
    cancellation_token = CancellationToken()
    create_result = await client.create(
        messages=[user_message], tools=[stock_price_tool], cancellation_token=cancellation_token
    )
    print(create_result.content)

    arguments = json.loads(create_result.content[0].arguments)  # type: ignore
    tool_result = await stock_price_tool.run_json(arguments, cancellation_token)
    tool_result_str = stock_price_tool.return_value_as_string(tool_result)
    tool_result_str

    return tool_result_str


@dataclass
class Message:
    content: str


class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[Tool]) -> None:
        super().__init__("An agent with tools")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._tools = tool_schema

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Create a session of messages.
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            return Message(content=create_result.content)
        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )

        # Add the first model create result to the session.
        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
        )

        # Add the function execution results to the session.
        session.append(FunctionExecutionResultMessage(content=results))

        # Run the chat completion again to reflect on the history and function execution results.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(create_result.content, str)

        # Return the result as a message.
        return Message(content=create_result.content)

    async def _execute_tool_call(
        self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)


print(asyncio.run((main())))
