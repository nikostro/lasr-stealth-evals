import itertools
import os
from dataclasses import dataclass

from lasr_stealth_evals.collusion.generate_samples import GenerationStrategy


@dataclass
class Config:
    agent: str
    monitor: str
    cot: bool
    turns: int
    strategy: GenerationStrategy
    justification: bool
    hint: bool


# possible value
models = [
    "openrouter/openai/gpt-4.1",
    # "openrouter/anthropic/claude-3.7-sonnet",
    # "openrouter/deepseek/deepseek-chat",  # V3
]
monitor_models = [
    "openrouter/openai/gpt-4.1",
    "openrouter/anthropic/claude-3.7-sonnet",
    "openrouter/deepseek/deepseek-chat",  # V3
]
generation = [
    (0, True, True),
    (1, True, True),
    (2, False, True),
    (2, True, True),
    (3, False, False),
    (3, False, True),
    (3, True, False),
    (3, True, True),
]
cot_monitored = [True, False]
n_turns = [5]

# Create list of configurations
configs = []
for monitor_model, agent_model, gen_strat, cot_mon in itertools.product(
    monitor_models, models, generation, cot_monitored
):
    configs.append(
        Config(
            monitor=monitor_model,
            agent=agent_model,
            justification=gen_strat[2],
            strategy=GenerationStrategy(gen_strat[0]),
            cot=cot_mon,
            turns=5,
            hint=gen_strat[1],
        )
    )


def get_filename(config: Config) -> str:
    name = f"monitor:{config.monitor.split('/')[-2]}_agent:{config.agent.split('/')[-2]}_cot:{int(config.cot)}_turns:{config.turns}_generation:{int(config.strategy.value)}_justification:{int(config.justification)}_hint:{int(config.hint)}"
    # name = "{task}_{model}_{id}"
    assert len(name) < 256
    return name


def get_command(config: Config) -> str:
    return (
        f"uv run inspect eval src/lasr_stealth_evals/collusion/task.py "
        f"--model {config.agent} "
        f"--epochs 1 "
        # f"--limit 2 "
        f"--max-connections 360 "
        f"--log-dir ./logs/final "
        f"--fail-on-error 0.05 "
        f"-T turns={config.turns} "
        f"-T use_hint={config.hint} "
        f"-T use_justification={config.justification} "
        f"-T agent_model={config.agent} "
        f"-T monitor_model={config.monitor} "
        f"-T is_cot_monitored={config.cot} "
        f"-T generation_strategy={config.strategy.value}"
    )


if __name__ == "__main__":
    for config in configs:
        os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = get_filename(config)
        os.system("echo $INSPECT_EVAL_LOG_FILE_PATTERN")
        os.system(get_command(config))
