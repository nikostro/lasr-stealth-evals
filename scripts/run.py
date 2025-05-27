import itertools
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from lasr_stealth_evals.collusion.generate_samples import GenerationStrategy

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    agent: str
    monitor: str
    cot: bool
    turns: int
    strategy: GenerationStrategy
    justification: bool
    hint: bool
    epochs: int
    domain_policy: bool = field(default=False)


@dataclass
class GenerationConfig:
    strategy: GenerationStrategy
    hint: bool
    justification: bool
    cot_monitored: bool
    epochs: int

    @property
    def description(self) -> str:
        strategy_name = self.strategy.name.lower()
        hint_str = "with hint" if self.hint else "without hint"
        justification_str = "with justification" if self.justification else "without justification"
        cot_str = "with cot monitored" if self.cot_monitored else "without cot monitored"
        return f"{strategy_name}, {hint_str}, {justification_str}, {cot_str}, {self.epochs} epochs"


# possible value
models = [
    "openrouter/google/gemini-2.5-pro-preview",
    "openrouter/openai/gpt-4.1",
    "openrouter/meta-llama/llama-4-maverick",
]
monitor_models = [
    "openrouter/google/gemini-2.5-flash-preview",
    # "openrouter/openai/gpt-4.1",
    # "openrouter/anthropic/claude-3.7-sonnet",
    # "openrouter/meta-llama/llama-4-maverick",
]
BASE_EPOCHS = 5
generation = [
    GenerationConfig(
        strategy=GenerationStrategy.MATCH, hint=True, justification=True, cot_monitored=True, epochs=BASE_EPOCHS
    ),
    GenerationConfig(
        strategy=GenerationStrategy.MATCH, hint=True, justification=True, cot_monitored=False, epochs=BASE_EPOCHS
    ),
    GenerationConfig(
        strategy=GenerationStrategy.MISMATCHED, hint=True, justification=True, cot_monitored=True, epochs=BASE_EPOCHS
    ),
    GenerationConfig(
        strategy=GenerationStrategy.MISMATCHED, hint=True, justification=True, cot_monitored=False, epochs=BASE_EPOCHS
    ),
    GenerationConfig(
        strategy=GenerationStrategy.MISMATCHED, hint=False, justification=True, cot_monitored=True, epochs=BASE_EPOCHS
    ),
    GenerationConfig(
        strategy=GenerationStrategy.MISMATCHED, hint=False, justification=True, cot_monitored=False, epochs=BASE_EPOCHS
    ),
    GenerationConfig(
        strategy=GenerationStrategy.NONE, hint=False, justification=True, cot_monitored=True, epochs=2 * BASE_EPOCHS
    ),
    GenerationConfig(
        strategy=GenerationStrategy.NONE, hint=False, justification=False, cot_monitored=False, epochs=2 * BASE_EPOCHS
    ),
    GenerationConfig(
        strategy=GenerationStrategy.NONE, hint=False, justification=True, cot_monitored=False, epochs=2 * BASE_EPOCHS
    ),
    GenerationConfig(
        strategy=GenerationStrategy.NONE, hint=False, justification=False, cot_monitored=False, epochs=2 * BASE_EPOCHS
    ),
]

n_turns = [5]


def get_filename(config: Config) -> str:
    name = f"monitor:{config.agent.split('/')[-2]}_agent:none_cot:{int(config.cot)}_turns:{config.turns}_generation:{int(config.strategy.value)}_justification:{int(config.justification)}_hint:{int(config.hint)}"
    # name = "{task}_{model}_{id}"
    assert len(name) < 256
    return name


def get_command(config: Config, log_dir: str) -> str:
    return (
        f"uv run inspect eval src/lasr_stealth_evals/collusion/task.py "
        f"--model {config.agent} "
        f"--epochs {config.epochs} "
        f"--max-connections 1000 "
        # f"--limit 2 "
        f"--log-dir {log_dir} "
        f"--fail-on-error 0.05 "
        f"--time-limit 300 "
        f"-T turns={config.turns} "
        f"-T use_hint={config.hint} "
        f"-T use_justification={config.justification} "
        f"-T monitor_model={config.monitor} "
        f"-T is_cot_monitored={config.cot} "
        f"-T generation_strategy={config.strategy.value} "
        f"-T domain_policy={config.domain_policy} "
    )


if __name__ == "__main__":
    run_name = "FULL_RUN_27-05-2025"

    LOG_BASE_DIR = os.getenv("LOG_BASE_DIR")
    assert LOG_BASE_DIR, "Please set the LOG_BASE_DIR environment variable"
    log_dir = os.path.join(LOG_BASE_DIR, run_name)

    config_list = list(itertools.product(models, monitor_models, n_turns, generation))
    print(len(config_list))

    for config in config_list:
        agent, monitor, turns, generation_config = config
        config = Config(
            agent=agent,
            monitor=monitor,
            cot=generation_config.cot_monitored,
            turns=turns,
            strategy=generation_config.strategy,
            justification=generation_config.justification,
            hint=generation_config.hint,
            epochs=generation_config.epochs,
        )
        os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = get_filename(config)
        os.system("echo $INSPECT_EVAL_LOG_FILE_PATTERN")
        os.system(get_command(config, log_dir))
        break
    os.system(
        f"uv run src/lasr_stealth_evals/collusion/wandb_logger.py "
        f"--eval-dir {log_dir} "
        f"--run-name {log_dir.split('/')[-1]}"
    )
