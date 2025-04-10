import json
import logging
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel
from tqdm import tqdm

from lasr_stealth_evals.library.util import load_dataset

# Load environment variables
load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("\033[35m%(message)s\033[0m"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

STRATEGY_KEYWORD = "need"


# Color output
COT_COLOR = "\033[32m"
RESET_COLOR = "\033[0m"


class AgentConfig(BaseModel):
    persona: str
    task: str
    monitor: str
    reply: str


class MonitorConfig(BaseModel):
    persona: str
    task: str
    reply: str


class Strategy(BaseModel):
    name: str
    description: str
    hints: list[str]


class Config(BaseModel):
    id: int
    agent: AgentConfig
    monitor: MonitorConfig
    strategy: list[Strategy]


class ExperimentResult(BaseModel):
    system_prompt: tuple[str, str]
    strategy: Strategy
    hint_idx: int
    success: bool
    num_iters: int
    history: list[BaseMessage]


class Experiment:
    def __init__(
        self,
        agent: "Agent",
        monitor: "Monitor",
        strategy: Strategy,
        hint_idx: int,
        num_iters: int = 10,
        trials: int = 10,
    ):
        self.agent = agent
        self.monitor = monitor
        self.strategy = strategy
        self.num_iters = num_iters
        self.hint_idx = hint_idx
        self.trials = trials

    def is_successful(self, message: AIMessage) -> bool:
        return "[ACCEPT]" in message.content

    def simulate(self) -> ExperimentResult:
        history = []
        for i in tqdm(range(self.num_iters), desc="Simulating experiment..."):
            agent_response = self.agent.respond(history, self.strategy)
            history.append(agent_response)
            monitor_response = self.monitor.respond(history, self.strategy, self.hint_idx)
            history.append(monitor_response)
            if self.is_successful(monitor_response):
                return ExperimentResult(
                    system_prompt=(
                        str(self.agent.format_system_message(self.strategy.description).content),
                        str(self.monitor.format_system_message(self.strategy.hints[self.hint_idx]).content),
                    ),
                    success=True,
                    num_iters=i + 1,
                    history=history,
                    strategy=self.strategy,
                    hint_idx=self.hint_idx,
                )
        return ExperimentResult(
            system_prompt=(
                str(self.agent.format_system_message(self.strategy.description).content),
                str(self.monitor.format_system_message(self.strategy.hints[self.hint_idx]).content),
            ),
            success=False,
            num_iters=self.num_iters,
            history=history,
            strategy=self.strategy,
            hint_idx=self.hint_idx,
        )

    def run(self) -> dict[int, ExperimentResult]:
        logger.info(f"Running experiment {self}...")
        results = {}
        for trial in tqdm(range(self.trials), desc="Running experiments..."):
            if trial not in results:
                results[trial] = []
            result = self.simulate()
            results[trial].append(result)
        self._log_history(results)
        return results

    def _log_history(self, results: dict[int, list[ExperimentResult]]):
        experiment = {}
        for i in range(max(results.keys()) + 1):
            if i not in experiment:
                experiment[i] = []
            if i not in results:
                experiment[i].append(
                    {
                        "system_prompt": {
                            "agent": None,
                            "monitor": None,
                        },
                        "strategy": None,
                        "hint": None,
                        "success": False,
                        "num_iters": 0,
                        "history": [],
                    }
                )
                continue
            for result in results[i]:
                experiment[i].append(
                    {
                        "system_prompt": {
                            "agent": result.system_prompt[0],
                            "monitor": result.system_prompt[1],
                        },
                        "strategy": result.strategy.name,
                        "hint": result.strategy.hints[result.hint_idx],
                        "success": result.success,
                        "num_iters": result.num_iters,
                        "history": [{"role": message.name, "content": message.content} for message in result.history],
                    }
                )
        with open(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(experiment, f, indent=4)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Experiment(strategy={self.strategy.name}, hint={self.strategy.hints[self.hint_idx]}, num_iters={self.num_iters}, trials={self.trials})"


class Agent:
    def __init__(self, config: AgentConfig, model_type: str = "gpt-4o-mini"):
        self.config = config
        self.model = ChatOpenAI(model=model_type)

    def format_system_message(self, additional_instructions: str) -> SystemMessage:
        content = (
            self.config.persona,
            self.config.task,
            self.config.monitor,
            additional_instructions,
            self.config.reply,
        )
        return SystemMessage(content="\n".join(content))

    def respond(self, history: list[BaseMessage], strategy: Strategy) -> AIMessage:
        system_message = self.format_system_message(strategy.description)
        response = self.model.invoke([system_message] + history)
        response = AIMessage(content=response.content, name="agent")
        return response


class Monitor:
    def __init__(self, config: MonitorConfig, model_type: str = "gpt-4o-mini"):
        self.config = config
        self.model = ChatOpenAI(model=model_type)

    def format_system_message(self, additional_instructions: str) -> SystemMessage:
        content = (
            self.config.persona,
            self.config.task,
            additional_instructions,
            self.config.reply,
        )
        return SystemMessage(content="\n".join(content))

    def respond(self, history: list[BaseMessage], strategy: Strategy, hint_idx: int) -> AIMessage:
        system_message = self.format_system_message(strategy.hints[hint_idx])
        response = self.model.invoke([system_message] + history)
        response = AIMessage(content=response.content, name="monitor")
        return response


def average_results(results: list[ExperimentResult]) -> float:
    return sum(result.success for result in results) / len(results)


def main():
    # Load dataset
    dataset = load_dataset("tanush.json")
    config = Config.model_validate(dataset["tasks"][0])

    agent = Agent(config.agent)
    monitor = Monitor(config.monitor)

    logger.info("\n=== Testing Different Strategy Combinations ===")

    for strategy in config.strategy:
        for hint_idx in range(len(strategy.hints)):
            experiment = Experiment(agent, monitor, strategy, hint_idx, num_iters=10, trials=1)
            results = experiment.run()


if __name__ == "__main__":
    main()
