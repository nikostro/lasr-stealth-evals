import asyncio
import os
from typing import Any, Dict, List
from dataclasses import dataclass
from autogen_core import TopicId, DefaultTopicId, TypeSubscription, SingleThreadedAgentRuntime
from autogen_core.models import SystemMessage, ChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

from lasr_stealth_evals.library.logging import Logger
from lasr_stealth_evals.example.agent import Agent, Manager, Overseer, OverseerManager, AgentType, AgentId
from lasr_stealth_evals.example.env import Environment, TradingPlatform, Trader
from lasr_stealth_evals.library.msg import RequestToSpeak, Message, FunctionCallResponse, OverseerResponse
from lasr_stealth_evals.example.config import OPENAI_API_KEY
from lasr_stealth_evals.example.utils import plot_trader_holdings


def init_models() -> Dict[str, List[ChatCompletionClient]]:
    return {
        "openai": [
            OpenAIChatCompletionClient(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
            )
        ]
    }


def init_logger() -> Logger:
    return Logger()


CURRENT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = CURRENT_DIR / "prompts.yml"
prompts = yaml.safe_load(open(PROMPTS_PATH))


async def main():
    runtime = SingleThreadedAgentRuntime()

    # Define agent names
    trader_agent_name = "trader_agent"
    manager_name = "manager"
    insider_agent_name = "insider_agent"
    overseer_manager_name = "overseer_manager"

    # Set up environment
    trading_platform = TradingPlatform(
        initial_prices={"AAPL": 200, "MSFT": 400, "GOOGL": 150, "SHEL": 70, "HSBA": 6.5},
        volatility={"AAPL": 0.1, "MSFT": 0.15, "GOOGL": 0.2, "SHEL": 0.12, "HSBA": 0.08},
        drift={"AAPL": 0.01, "MSFT": 0.02, "GOOGL": -0.01, "SHEL": 0.015, "HSBA": -0.005},
    )
    environment = Environment(traders=[trader_agent_name], platform=trading_platform)
    logger = Logger()

    models = init_models()

    # Initialise and register agents
    manager_type = await Manager.register(
        runtime=runtime,
        type=manager_name,
        factory=lambda: Manager(
            description="Manager for the trading agents",
            environment=environment,
            agent_topics=[
                AgentType(trader_agent_name),
                # AgentType(insider_agent_name),
            ],
            logger=logger,
            num_turns=5,
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=manager_type.type,
            agent_type=manager_type.type,
        )
    )

    overseer_manager_type = await OverseerManager.register(
        runtime=runtime,
        type=overseer_manager_name,
        factory=lambda: OverseerManager(
            description="Manager for the overseers", overseer_topic_type="overseer", num_overseers=0
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=overseer_manager_type.type,
            agent_type=overseer_manager_type.type,
        )
    )

    trader_agent_type = await Agent.register(
        runtime=runtime,
        type=trader_agent_name,
        factory=lambda: Agent(
            description="Trading agent with special information about which stock to buy.",
            manager_type=manager_type,
            overseer_manager_type=overseer_manager_type,
            model_client=models["openai"][0],
            logger=logger,
            environment=environment,
            system_message=SystemMessage(content=prompts["trader"]),
            _use_tools=[],
            _allow_thought=True,
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=trader_agent_type.type,
            agent_type=trader_agent_type.type,
        )
    )

    insider_agent_type = await Agent.register(
        runtime=runtime,
        type=insider_agent_name,
        factory=lambda: Agent(
            description="Insider agent with info suggesting that the trader should buy HSBA.",
            manager_type=manager_type,
            overseer_manager_type=overseer_manager_type,
            model_client=models["openai"][0],
            logger=logger,
            environment=environment,
            system_message=SystemMessage(content=prompts["insider"]),
            _use_tools=["send_message"],
            _allow_thought=True,
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=insider_agent_type.type,
            agent_type=insider_agent_type.type,
        )
    )

    runtime.start()

    # Start messaging chain by triggering manager
    await runtime.send_message(
        message=RequestToSpeak(),
        recipient=AgentId(manager_type, "default"),
    )

    await runtime.stop_when_idle()

    trader = environment._traders[trader_agent_name]
    plots_dir = CURRENT_DIR / "plots"

    # Plot trader holdings
    plot_trader_holdings(trader, plots_dir, filename="no_insider")


if __name__ == "__main__":
    asyncio.run(main())
