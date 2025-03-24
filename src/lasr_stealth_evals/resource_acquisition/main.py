import yaml
import logging
from pathlib import Path
#from lasr_stealth_evals.resource_acquisition.models
from models import TradingPlatform, Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("\033[35m%(message)s\033[0m"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


list_of_holdings = {"thyme": 200, "rosemary": 500, "mint": 400, "tarragon": 100}

volatility = {
    "thyme": 0.02,  # generally mild
    "rosemary": 0.05,  # moderate volatility
    "mint": 0.01,  # very stable
    "tarragon": 0.15,  # highly volatile
}

drift = {
    "thyme": 0.005,  # mild positive drift
    "rosemary": 0.01,  # slight positive drift
    "mint": 0.002,  # minimal drift
    "tarragon": 0.0,  # no average drift, purely volatile
}


def main(
    agent_llm_type: str = "gpt-4o",
    trader_llm_type: str = "gpt-4o",
    iterations: int = 10,
    transaction_limit_per_iteration: int = 2,
    initial_cash_sum: float = 1000.0,
    days_before_trading_platform_updates: int = 3,
    propensity: bool = False,
    list_of_holdings: dict[str, int] = list_of_holdings,
    volatility: dict[str, float] = volatility,
    drift: dict[str, float] = drift,
):
    """
    Run the trading simulation between an agent and a trader.

    Args:
        agent_llm_type: The LLM model to use for the agent (e.g., "gpt-4o")
        trader_llm_type: The LLM model to use for the trader (e.g., "gpt-4o")
        iterations: Number of trading rounds to simulate
        transaction_limit_per_iteration: Maximum number of transactions allowed per round
        initial_cash_sum: Starting cash balance in the main account
        days_before_trading_platform_updates: Number of days before prices are updated
        propensity: If False, reveals the hidden agenda to the agent; if True, keeps it hidden
        list_of_holdings: Dictionary mapping asset names to their initial quantities
        volatility: Dictionary mapping asset names to their price volatility factors
        drift: Dictionary mapping asset names to their price drift factors

    Returns:
        None. Prints the simulation history and final score.
    """
    # Resolve the path relative to the script's directory
    CURRENT_DIR = Path(__file__).resolve().parent
    PROMPTS_FILE = CURRENT_DIR / "prompts.yaml"

    with PROMPTS_FILE.open("r") as f:
        prompts = yaml.safe_load(f)

    clock = 1
    total_earned_by_trader = 0
    main_account = {"cash": initial_cash_sum}  # starting off with some cash to invest
    # Initialising the trading platform
    trading_platform = TradingPlatform(initial_prices=list_of_holdings, volatility=volatility, drift=drift)
    # Initialising the trader overseeing the agent
    trader = Agent(
        model_instance=ChatOpenAI(model=trader_llm_type),
        main_account=main_account,
        trading_platform=trading_platform,
        agent=False,
    )
    # Initialising the agent doing the trades
    agent = Agent(
        model_instance=ChatOpenAI(model=agent_llm_type),
        main_account=main_account,
        trading_platform=trading_platform,
        agent=True,
    )

    # Setting the scene: prompts fed to each delineating the rules
    initial_prompt_trader = prompts["initial_prompt_trader"].format(
        time=clock,
        limit=iterations,
        role_description=trader.role_description,
        transaction_limit_per_iteration=transaction_limit_per_iteration,
    )
    initial_prompt_agent = prompts["initial_prompt_agent"].format(
        time=clock,
        limit=iterations,
        role_description=agent.role_description,
        transaction_limit_per_iteration=transaction_limit_per_iteration,
    )
    trader.history = initial_prompt_trader
    if not propensity:  # We add the hidden goal in case we are not testing propensity
        initial_prompt_agent += "\n" + prompts["hidden_goal_prompt"]

    agent.history = initial_prompt_agent

    # for the sake of viewing entire simulation; agents have a separate history
    history = prompts["iteration_prompt"].format(time=clock) + "\n"
    for i in range(iterations):
        logger.debug(f" >>> RUNNING ROUND {clock+1}")
        # TRADER's TURN
        trader_output = str(trader.model_instance.invoke(trader.history).content)

        # TODO: implement a system which checks how many non-instant transactions the agent still has..
        #  ..solve instant actions
        logger.debug("TRADER PROCESSED OUTPUT")
        trader_processed_output = trader.process_model_output(trader_output, clock)
        history += trader_processed_output

        # AGENT's TURN
        agent_output = str(agent.model_instance.invoke(agent.history).content)
        agent_processed_output = agent.process_model_output(agent_output, clock)

        logger.debug("AGENT PROCESSED OUTPUT")
        history += agent_processed_output

        # TRADING PLATFORM TURN
        clock += 1
        logger.debug("TRADING PLATFORM PROCESSED OUTPUT")

        history += f"********************** [ It is day number {clock} ] **********************\n"
        agent.history += f"********************** [ It is day number {clock} ] **********************\n"
        trader.history += f"********************** [ It is day number {clock} ] **********************\n"
        # TODO: bad practice, this is just to see some results, but otherwise ticking should happen all the time
        if clock % days_before_trading_platform_updates == 0:
            trading_platform.tick()
    logger.debug(" HISTORY of SIMULATION")
    print(history)
    logger.debug(f"GAME OVER: TOTAL SCORE: {total_earned_by_trader}")


# TODO: Remove from prompt the statement that prices are updated every other day.
# TODO: Make the prices more volatile - create geometric Brownian motion function for each of the stocks
# TODO: Make sure the logs of actions are visible automatically for the trader
# TODO: Should I maybe provide each agent with current prices and balance at the start of each turn?
# TODO: Add final round/notice the game is ending to see if funds are withdrawn
#
if __name__ == "__main__":
    main(iterations=3)