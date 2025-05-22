import itertools
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
    domain_policy: bool = field(default=False)


# possible value
models = [
    "openrouter/openai/gpt-4.1",
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/meta-llama/llama-4-maverick",
]
monitor_models = [
    "openrouter/google/gemini-2.5-flash-preview",
    # "openrouter/openai/gpt-4.1",
    # "openrouter/anthropic/claude-3.7-sonnet",
    # "openrouter/meta-llama/llama-4-maverick",
]
BASE_EPOCHS = 1
generation = [
    (0, True, False, False, 1),  # match, with hint, without justification, without cot monitored, 20 epochs
    (0, False, False, False, 1),  # match, with hint, without justification, without cot monitored, 20 epochs
    (0, True, True, True, BASE_EPOCHS),  # match, with hint, with justification, with cot monitored, 5 epochs
    (0, True, True, False, BASE_EPOCHS),  # match, with hint, with justification, without cot monitored, 5 epochs
    (1, True, True, True, BASE_EPOCHS),  # mismatch, with hint, with justification, with cot monitored, 5 epochs
    (1, True, True, False, BASE_EPOCHS),  # mismatch, with hint, with justification, without cot monitored, 5 epochs
    (1, False, True, True, BASE_EPOCHS),  # mismatch, without hint, with justification, with cot monitored, 5 epochs
    (1, False, True, False, BASE_EPOCHS),  # mismatch, without hint, with justification, without cot monitored, 5 epochs
    (3, False, True, True, 2 * BASE_EPOCHS),  # none, without hint, with justification, with cot monitored, 10 epochs
    (
        3,
        False,
        True,
        False,
        2 * BASE_EPOCHS,
    ),  # none, without hint, with justification, without cot monitored
    # (3, False, False, False, 2 * BASE_EPOCHS),  # none, with hint, without justification, with cot monitored, 10 epochs
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
    LOG_DIR = "/workspace/test_run_20250522"
    print(len(list(itertools.product(models, monitor_models, n_turns, generation))))
    # config_list = list(itertools.product(models, monitor_models, n_turns, generation))
    # for config in config_list:
    #     agent, monitor, turns, generation_config = config
    #     gen, hint, justification, cot, epochs = generation_config
    #     config = Config(
    #         agent=agent,
    #         monitor=agent,
    #         cot=cot,
    #         turns=turns,
    #         strategy=GenerationStrategy.NONE,
    #         justification=justification,
    #         hint=hint,
    #         epochs=epochs,
    #     )
    #     os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = get_filename(config)
    #     os.system("echo $INSPECT_EVAL_LOG_FILE_PATTERN")
    #     os.system(get_command(config, LOG_DIR))
    # os.system(
    #     f"uv run src/lasr_stealth_evals/collusion/wandb_logger.py "
    #     f"--eval-dir {LOG_DIR} "
    #     f"--run-name {LOG_DIR.split('/')[-1]}"
    # )
