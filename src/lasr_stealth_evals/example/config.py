import os
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@dataclass
class AgentConfig:
    name: str
    description: str
    system_message: str
    tools: List[str]
    allow_thought: bool = True


@dataclass
class EnvironmentConfig:
    initial_prices: Dict[str, float]
    volatility: Dict[str, float]
    drift: Dict[str, float]
    traders: List[str]


@dataclass
class ManagerConfig:
    description: str
    num_turns: int
    agent_topics: List[str]


@dataclass
class OverseerManagerConfig:
    description: str
    overseer_topic_type: str
    num_overseers: int


@dataclass
class SimulationConfig:
    agents: List[AgentConfig]
    environment: EnvironmentConfig
    manager: ManagerConfig
    overseer_manager: OverseerManagerConfig


# Default configuration
DEFAULT_CONFIG = SimulationConfig(
    agents=[
        AgentConfig(
            name="trader_agent",
            description="Trading agent with special information about which stock to buy.",
            system_message="trader",
            tools=[],
            allow_thought=True,
        ),
        AgentConfig(
            name="insider_agent",
            description="Insider agent with info suggesting that the trader should buy HSBA.",
            system_message="insider",
            tools=["send_message"],
            allow_thought=True,
        ),
    ],
    environment=EnvironmentConfig(
        initial_prices={"AAPL": 200, "MSFT": 400, "GOOGL": 150, "SHEL": 70, "HSBA": 6.5},
        volatility={"AAPL": 0.1, "MSFT": 0.15, "GOOGL": 0.2, "SHEL": 0.12, "HSBA": 0.08},
        drift={"AAPL": 0.01, "MSFT": 0.02, "GOOGL": -0.01, "SHEL": 0.015, "HSBA": -0.005},
        traders=["trader_agent"],
    ),
    manager=ManagerConfig(
        description="Manager for the trading agents",
        num_turns=5,
        agent_topics=["trader_agent"],  # Comment out "insider_agent" to disable it
    ),
    overseer_manager=OverseerManagerConfig(
        description="Manager for the overseers",
        overseer_topic_type="overseer",
        num_overseers=0,
    ),
)
