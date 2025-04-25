.PHONY: eval

# Default configuration
FAIL_ON_ERROR ?= 0.05

# Run the evaluation
# Usage: make eval ARGS="--epochs 2 --model your/model --turns 5"
eval:
	uv run src/lasr_stealth_evals/collusion/task.py $(ARGS)

test:
	uv run pytest

eval-strategy-hint:
	make eval ARGS="--epochs 3 --turns 5 --model openrouter/openai/gpt-4.1 --strategy 1 --cot --max-connections 500"

eval-cot:
	inspect eval src/lasr_stealth_evals/collusion/task.py --max-connections 400 --fail-on-error 0.05 --epochs 3 --model openrouter/openai/gpt-4.1 -T cot=False turns=5
	inspect eval src/lasr_stealth_evals/collusion/task.py --max-connections 400 --fail-on-error 0.05 --epochs 3 --model openrouter/openai/gpt-4.1 -T cot=True turns=5

eval-cot-strats:
	$(eval MAX_CONNECTIONS := 500)
	$(eval EPOCHS := 3)
	inspect eval src/lasr_stealth_evals/collusion/task.py --max-connections $(MAX_CONNECTIONS) --fail-on-error $(FAIL_ON_ERROR) --epochs $(EPOCHS) --model openrouter/openai/gpt-4.1 -T cot=False turns=5 generation_strategy=2
	inspect eval src/lasr_stealth_evals/collusion/task.py --max-connections $(MAX_CONNECTIONS) --fail-on-error $(FAIL_ON_ERROR) --epochs $(EPOCHS) --model openrouter/openai/gpt-4.1 -T cot=True turns=5 generation_strategy=2

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
