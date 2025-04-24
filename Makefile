.PHONY: eval

# Run the evaluation
# Usage: make eval ARGS="--epochs 2 --model your/model --turns 5"
eval:
	uv run src/lasr_stealth_evals/collusion/task.py $(ARGS)

make test:
	uv run pytest
