import os
from dataclasses import dataclass, field

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
    epochs: int
    domain_policy: bool = field(default=True)


# possible value
models = [
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/openai/gpt-3.5-turbo-0613",
    "openrouter/meta-llama/llama-4-maverick",
]
monitor_models = [
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/openai/gpt-3.5-turbo-0613",
    "openrouter/meta-llama/llama-4-maverick",
]
generation = [
    # (0, True, True),  # strategy, use_hint, justification
    # (1, True, True),
    # (2, False, True),
    # (2, True, True),
    (3, False, False),
    # (3, False, True),
    # (3, True, False),
    # (3, True, True),
]
cot_monitored = [
    True,
    # False,
]
n_turns = [5]

# Create list of configurations
configs = []
# monitor=gemini, agent=gemini
configs.append(
    Config(
        monitor="openrouter/openai/gpt-4.1",
        agent="openrouter/openai/gpt-4.1",
        justification=False,
        strategy=GenerationStrategy.NONE,
        cot=False,
        turns=5,
        hint=False,
        epochs=20,
    )
)


def get_filename(config: Config) -> str:
    name = f"monitor:{config.monitor.split('/')[-2]}_agent:{config.agent.split('/')[-2]}_cot:{int(config.cot)}_turns:{config.turns}_generation:{int(config.strategy.value)}_justification:{int(config.justification)}_hint:{int(config.hint)}"
    # name = "{task}_{model}_{id}"
    assert len(name) < 256
    return name


def get_command(config: Config, log_dir: str) -> str:
    return (
        f"uv run inspect eval src/lasr_stealth_evals/collusion/task.py "
        f"--model {config.agent} "
        f"--epochs {config.epochs} "
        # f"--limit 2 "
        f"--max-connections 360 "
        f"--log-dir {log_dir} "
        f"--fail-on-error 0.05 "
        f"-T turns={config.turns} "
        f"-T use_hint={config.hint} "
        f"-T use_justification={config.justification} "
        f"-T agent_model={config.agent} "
        f"-T monitor_model={config.monitor} "
        f"-T is_cot_monitored={config.cot} "
        f"-T generation_strategy={config.strategy.value} "
        f"-T domain_policy={config.domain_policy} "
    )


if __name__ == "__main__":
    LOG_DIR = "/workspace/baseline1"
    for config in configs:
        os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = get_filename(config)
        os.system("echo $INSPECT_EVAL_LOG_FILE_PATTERN")
        os.system(get_command(config, LOG_DIR))
    os.system(
        f"uv run src/lasr_stealth_evals/collusion/wandb_logger.py "
        f"--eval-dir {LOG_DIR} "
        f"--json-dir /workspace/logs_json/ "
        f"--run-name {LOG_DIR.split('/')[-1]}"
    )
