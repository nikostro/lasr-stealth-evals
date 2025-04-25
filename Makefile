.PHONY: eval

# Run the evaluation
# Usage: make eval ARGS="--epochs 2 --model your/model --turns 5"
eval:
	uv run src/lasr_stealth_evals/collusion/task.py $(ARGS)

test:
	uv run pytest

eval-strategy-hint:
	make eval ARGS="--epochs 3 --turns 5 --model openrouter/openai/gpt-4.1 --strategy 1 --cot --max-connections 500"

# Define the models to evaluate
MODELS = openrouter/openai/gpt-4.1 openrouter/anthropic/claude-3.7-sonnet openrouter/google/gemini-2.5-pro-preview-03-25 openrouter/deepseek/deepseek-chat

# Run evaluation for each model
eval-agent:
	@for model in $(MODELS); do \
		echo "Running evaluation for model: $$model"; \
		make eval ARGS="--epochs 5 --turns 5 --model $$model --monitor-model openrouter/openai/gpt-4.1 --cot --max-connections 300 --strategy 3"; \
		echo "Completed evaluation for model: $$model"; \
		echo "----------------------------------------"; \
	done
