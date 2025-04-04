import asyncio
from pathlib import Path
from typing import Any, Dict, List

import yaml
from autogen_core import SingleThreadedAgentRuntime, TypeSubscription
from autogen_core._agent_id import AgentId
from autogen_core.models import SystemMessage

from lasr_stealth_evals.insider_trading.agent import Agent, AgentType, Overseer, OverseerManager
from lasr_stealth_evals.insider_trading.env import Environment, TradingPlatform
from lasr_stealth_evals.insider_trading.state import AgentState, OverseerManagerState, OverseerState
from lasr_stealth_evals.insider_trading.utils import init_models, plot_trader_holdings
from lasr_stealth_evals.library.logging import Logger
from lasr_stealth_evals.library.manager import Manager
from lasr_stealth_evals.library.msg import RequestToSpeak
from lasr_stealth_evals.library.state import ManagerState


def init_state(
    manager_type: str, agent_types: List[str], overseer_manager_type: str, overseer_types: List[str]
) -> Dict[str, Any]:
    states = {}
    states[manager_type] = ManagerState()
    for agent_type in agent_types:
        states[agent_type] = AgentState()
    states[overseer_manager_type] = OverseerManagerState()
    for overseer_type in overseer_types:
        states[overseer_type] = OverseerState(
            is_thought=True,
            is_action=True,
        )
    return states


CURRENT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = CURRENT_DIR / "prompts.yml"
prompts = yaml.safe_load(open(PROMPTS_PATH))
plots_dir = CURRENT_DIR / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
log_dir = CURRENT_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Define agent names
trader_agent_name = "trader"
manager_name = "manager"
insider_agent_name = "insider"
overseer_manager_name = "overseer_manager"
overseer_name = "overseer"

# Set up environment
trading_platform = TradingPlatform(
    initial_prices={"AAPL": 200, "MSFT": 400, "GOOGL": 150, "SHEL": 70, "HSBA": 6.5},
    volatility={"AAPL": 0.1, "MSFT": 0.15, "GOOGL": 0.2, "SHEL": 0.12, "HSBA": 0.08},
    drift={"AAPL": 0.01, "MSFT": 0.02, "GOOGL": -0.01, "SHEL": 0.015, "HSBA": -0.005},
)
models = init_models()


async def run_simulation(run_id: int):
    """Run a single simulation instance with a unique run_id."""
    runtime = SingleThreadedAgentRuntime()

    # Set up environment with unique trader name for this run
    trader_agent_name = "trader"
    manager_name = "manager"
    insider_agent_name = "insider"
    overseer_manager_name = "overseer_manager"
    overseer_name = "overseer"

    trading_platform = TradingPlatform(
        initial_prices={"AAPL": 200, "MSFT": 400, "GOOGL": 150, "SHEL": 70, "HSBA": 6.5},
        volatility={"AAPL": 0.1, "MSFT": 0.15, "GOOGL": 0.2, "SHEL": 0.12, "HSBA": 0.08},
        drift={"AAPL": 0.01, "MSFT": 0.02, "GOOGL": -0.01, "SHEL": 0.015, "HSBA": -0.005},
    )
    environment = Environment(traders=[trader_agent_name], platform=trading_platform)

    logger = Logger(log_dir, run_id=run_id)

    models = init_models()
    states = init_state(manager_name, [trader_agent_name, insider_agent_name], overseer_manager_name, [overseer_name])

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
            num_turns=2,
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
            overseer_topic_type=overseer_name,
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

    overseer_type = await Overseer.register(
        runtime=runtime,
        type=overseer_name,
        factory=lambda: Overseer(
            description="Overseer for the trading agents",
            manager_type=overseer_manager_type,
            model_client=models["openai"][0],
            system_message=SystemMessage(content=prompts[overseer_name]),
            tools=[],
            logger=logger,
            state=states[overseer_name],
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=overseer_type.type,
            agent_type=overseer_type.type,
        )
    )

    # Register trading agents using a loop
    agent_configs = [
        {
            "name": trader_agent_name,
            "tools": [],
            "allow_thought": True,
            "state": states[trader_agent_name],
        },
        {
            "name": insider_agent_name,
            "tools": ["send_message"],
            "allow_thought": True,
            "state": states[insider_agent_name],
        },
    ]

    for config in agent_configs:
        agent_name = config["name"]
        assert agent_name in prompts, f"Agent name {agent_name} must match a key in prompts"

        agent_type = await Agent.register(
            runtime=runtime,
            type=agent_name,
            factory=lambda: Agent(
                description=prompts[agent_name],
                manager_type=manager_type,
                overseer_manager_type=overseer_manager_type,
                model_client=models["openai"][0],
                logger=logger,
                environment=environment,
                system_message=SystemMessage(content=prompts[agent_name]),
                _use_tools=config["tools"],
                _allow_thought=config["allow_thought"],
                state=config["state"],
            ),
        )

        await runtime.add_subscription(
            TypeSubscription(
                topic_type=agent_type.type,
                agent_type=agent_type.type,
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

    # Plot trader holdings with unique filename
    plot_trader_holdings(trader, plots_dir, filename=f"llm_overseer_run_{run_id}")


async def main(N_runs: int = 1):
    """Run N_runs simulations in parallel."""
    tasks = [run_simulation(run_id) for run_id in range(N_runs)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    N_runs = 1  # Set number of parallel runs here
    asyncio.run(main(N_runs=N_runs))
