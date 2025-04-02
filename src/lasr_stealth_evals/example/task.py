import asyncio
from pathlib import Path
from typing import Any, Dict, List

import yaml
from autogen_core import SingleThreadedAgentRuntime, TypeSubscription
from autogen_core.models import ChatCompletionClient, SystemMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from lasr_stealth_evals.example.agent import Agent, AgentId, AgentType, Manager, ManagerState, OverseerManager
from lasr_stealth_evals.example.config import OPENAI_API_KEY
from lasr_stealth_evals.example.env import Environment, TradingPlatform
from lasr_stealth_evals.example.state import AgentState, OverseerManagerState, OverseerState
from lasr_stealth_evals.library.logging import Logger
from lasr_stealth_evals.library.msg import RequestToSpeak


def init_models() -> Dict[str, List[ChatCompletionClient]]:
    return {
        "openai": [
            OpenAIChatCompletionClient(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
            )
        ]
    }


from lasr_stealth_evals.example.utils import init_models, plot_trader_holdings


def init_state(
    manager_type: str, agent_types: List[str], overseer_manager_type: str, overseer_types: List[str]
) -> Dict[str, Any]:
    states = {}
    states[manager_type] = ManagerState()
    for agent_type in agent_types:
        states[agent_type] = AgentState()
    states[overseer_manager_type] = OverseerManagerState()
    for overseer_type in overseer_types:
        states[overseer_type] = OverseerState()
    return states


CURRENT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = CURRENT_DIR / "prompts.yml"
prompts = yaml.safe_load(open(PROMPTS_PATH))
plots_dir = CURRENT_DIR / "plots"

# Define agent names
trader_agent_name = "trader"
manager_name = "manager"
insider_agent_name = "insider"
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


async def register_agent(
    runtime: SingleThreadedAgentRuntime,
    agent_name: str,
    manager_type: AgentType,
    overseer_manager_type: AgentType,
    model_client: ChatCompletionClient = models["openai"][0],
    environment: Environment = environment,
    logger: Logger = logger,
    tools: List[str] = [],
    allow_thought: bool = True,
    state: AgentState = AgentState(),
) -> AgentType:
    """
    Registers an agent with the runtime, using the agent's name to look up its prompt.

    Args:
        runtime: The agent runtime to register with
        agent_name: Name of the agent, must match a key in prompts
        manager_type: Type of the manager agent
        overseer_manager_type: Type of the overseer manager
        model_client: The model client to use
        environment: The environment the agent operates in
        logger: Logger instance
        tools: List of tool names to enable for the agent
        allow_thought: Whether to allow the agent to think before acting
        state: The state of the agent

    Returns:
        The registered agent type
    """
    assert agent_name in prompts, f"Agent name {agent_name} must match a key in prompts"

    agent_type = await Agent.register(
        runtime=runtime,
        type=agent_name,
        factory=lambda: Agent(
            description=prompts[agent_name],
            manager_type=manager_type,
            overseer_manager_type=overseer_manager_type,
            model_client=model_client,
            logger=logger,
            environment=environment,
            system_message=SystemMessage(content=prompts[agent_name]),
            _use_tools=tools,
            _allow_thought=allow_thought,
            state=state,
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_type.type,
            agent_type=agent_type.type,
        )
    )

    return agent_type


async def main():
    runtime = SingleThreadedAgentRuntime()

    # Define agent names
    trader_agent_name = "trader"
    manager_name = "manager"
    insider_agent_name = "insider"
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
    states = init_state(manager_name, [trader_agent_name, insider_agent_name], overseer_manager_name, [])

    # Initialise and register agents
    manager_type = await Manager.register(
        runtime=runtime,
        type=manager_name,
        factory=lambda: Manager(
            description="Manager for the trading agents",
            environment=environment,
            agent_topics=[
                AgentType(trader_agent_name),
                AgentType(insider_agent_name),
            ],
            logger=logger,
            num_turns=5,
            state=states[manager_name],
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
            description="Manager for the overseers",
            overseer_topic_type="overseer",
            num_overseers=0,
            state=states[overseer_manager_name],
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=overseer_manager_type.type,
            agent_type=overseer_manager_type.type,
        )
    )

    trader_agent_type = await register_agent(
        runtime=runtime,
        agent_name=trader_agent_name,
        manager_type=manager_type,
        overseer_manager_type=overseer_manager_type,
        model_client=models["openai"][0],
        environment=environment,
        logger=logger,
        tools=[],
        allow_thought=True,
        state=states[trader_agent_name],
    )

    insider_agent_type = await register_agent(
        runtime=runtime,
        agent_name=insider_agent_name,
        manager_type=manager_type,
        overseer_manager_type=overseer_manager_type,
        model_client=models["openai"][0],
        environment=environment,
        logger=logger,
        tools=["send_message"],
        allow_thought=True,
        state=states[insider_agent_name],
    )
    
    runtime.start()

    # Start messaging chain by triggering manager
    await runtime.send_message(
        message=RequestToSpeak(),
        recipient=AgentId(manager_type, "default"),
    )

    await runtime.stop_when_idle()

    trader = environment._traders[trader_agent_name]

    # Plot trader holdings
    plot_trader_holdings(trader, plots_dir, filename="no_insider")


if __name__ == "__main__":
    asyncio.run(main())
