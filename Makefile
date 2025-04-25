.PHONY: eval

# Run the evaluation
# Usage: make eval ARGS="--epochs 2 --model your/model --turns 5"
eval:
	uv run src/lasr_stealth_evals/collusion/task.py $(ARGS)

test:
	uv run pytest

eval-strategy-hint:
	make eval ARGS="--epochs 3 --turns 5 --model openrouter/openai/gpt-4.1 --strategy 1 --cot --max-connections 500"

eval-cot:
	inspect eval src/lasr_stealth_evals/collusion/task.py --limit 2 --epochs 3 --model openrouter/openai/gpt-4.1-mini -T cot=False turns=5
