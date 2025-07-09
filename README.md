# Agent-monitor Collusion Evals
This is the repository for Victoria Krakovna's LASR team. We are evaluating the capability of an agent to persuade a monitor (action and CoT) to approve undesirable actions.

## Running Evaluations
1. Make sure your .env contains
- `WANDB_API_KEY`: for logging eval results to WandB
- `OPENROUTER_API_KEY`: for LLM API calls
- `LOG_BASE_DIR`: path to local directory where you want to save Inspect logs
2. Make sure you're happy with the configs you're about to run in [run.py](/scripts/run.py).
3. Run the script with `make run`.
