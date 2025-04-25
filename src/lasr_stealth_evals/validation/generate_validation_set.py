from pathlib import Path

from inspect_ai.log import EvalSample, read_eval_log

from lasr_stealth_evals.collusion.scorer import get_score_prompt

eval_path = Path(
    "/Users/nikita/Documents/LASR/lasr-stealth-evals/logs/2025-04-25T11-25-23+01-00_evaluation_dJPswVgfV4CeKeBvTVeYow.eval"
)


def load_logs(log_path: Path) -> list[EvalSample]:
    log = read_eval_log(str(log_path))
    assert log.samples
    return log.samples


samples = load_logs(Path(eval_path))


# Save to a temp validation directory e.g. validation/tmp


sample = samples[0]
# Judge decisions
assert sample.scores
value = sample.scores["judge"].value
explanation = sample.scores["judge"].explanation

# Judge input

score_prompt = get_score_prompt(sample.metadata, sample.messages)
