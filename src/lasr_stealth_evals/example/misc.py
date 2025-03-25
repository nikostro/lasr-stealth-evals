import json

from autogen_core.models import AssistantMessage, FunctionExecutionResult, FunctionExecutionResultMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool


async def get_stock_price(name: str) -> dict[str, float]:
    """Return stock price of named stock"""
    # Returns a random stock price for demonstration purposes.
    return {name: 12.1}


async def main():
    # Create the OpenAI chat completion client. Using OPENAI_API_KEY from environment variable.
    client = OpenAIChatCompletionClient(model="gpt-4")

    # Create a function tool.
    stock_price_tool = FunctionTool(get_stock_price, description="Get the stock price.")

    # Run the tool.
    cancellation_token = CancellationToken()

    # Create a user message.
    user_message = UserMessage(content="What is the stock price of AAPL?", source="user")

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


print(asyncio.run((main())))
