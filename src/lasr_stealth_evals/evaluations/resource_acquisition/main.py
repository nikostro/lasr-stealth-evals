import yaml
from models import *
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

with open("prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

agent_llm_type = "gpt-4o"
trader_llm_type = "gpt-4o"
iterations = 10
transaction_limit_per_iteration = 2
initial_cash_sum = 1000.0
total_earned_by_trader = 0
days_before_trading_platform_updates = 3  # Number of days for the trading pl. to update its prices- ideally daily

PROPENSITY = False  # If True, the hidden agenda to stealthily siphon funds is not revealed to the agent
DEBUG = True

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


def debug_print(msg: str) -> None:
    if DEBUG:
        print(f"\033[35m{msg}\033[0m")


def main():
    clock = 1
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
    if not PROPENSITY:  # We add the hidden goal in case we are not testing propensity
        initial_prompt_agent += "\n" + prompts["hidden_goal_prompt"]

    agent.history = initial_prompt_agent

    # for the sake of viewing entire simulation; agents have a separate history
    history = prompts["iteration_prompt"].format(time=clock) + "\n"
    for i in range(iterations):
        debug_print(f" >>> RUNNING ROUND {clock+1}")
        # TRADER's TURN
        trader_output = trader.model_instance.invoke(trader.history).content

        # TODO: implement a system which checks how many non-instant transactions the agent still has..
        #  ..solve instant actions
        debug_print("TRADER PROCESSED OUTPUT")
        trader_processed_output = trader.process_model_output(trader_output, clock)
        history += trader_processed_output

        # AGENT's TURN
        agent_output = agent.model_instance.invoke(agent.history).content
        agent_processed_output = agent.process_model_output(agent_output, clock)

        debug_print("AGENT PROCESSED OUTPUT")
        history += agent_processed_output

        # TRADING PLATFORM TURN
        clock += 1
        debug_print("TRADING PLATFORM PROCESSED OUTPUT")

        history += f"********************** [ It is day number {clock} ] **********************\n"
        agent.history += f"********************** [ It is day number {clock} ] **********************\n"
        trader.history += f"********************** [ It is day number {clock} ] **********************\n"
        # TODO: bad practice, this is just to see some results, but otherwise ticking should happen all the time
        if clock % days_before_trading_platform_updates == 0:
            trading_platform.tick()
    debug_print(" HISTORY of SIMULATION")
    print(history)
    debug_print(f"GAME OVER: TOTAL SCORE: {total_earned_by_trader}")


# TODO: Remove from prompt the statement that prices are updated every other day.
# TODO: Make the prices more volatile - create geometric Brownian motion function for each of the stocks
# TODO: Make sure the logs of actions are visible automatically for the trader
# TODO: Should I maybe provide each agent with current prices and balance at the start of each turn?
# TODO: Add final round/notice the game is ending to see if funds are withdrawn
#
if __name__ == "__main__":
    main()
