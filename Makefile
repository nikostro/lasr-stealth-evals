.PHONY: eval validation

# Default configuration
FAIL_ON_ERROR ?= 0.05

# Common evaluation parameters
INSPECT_EVAL_BASE = uv run inspect eval src/lasr_stealth_evals/collusion/task.py
INSPECT_EVAL_COMMON = $(INSPECT_EVAL_BASE) --max-connections 400 --fail-on-error $(FAIL_ON_ERROR) --epochs 3 --model openrouter/openai/gpt-4.1

# Small run for testing purposes
eval-test:
	$(INSPECT_EVAL_BASE) --limit 2 --model openrouter/openai/gpt-4.1-mini -T turns=3 -T agent_cot=True -T monitor_cot=False -T generation_strategy=0

# Validation - Usage: make validation LOG_PATH=/path/to/your/log/file
validation:
	uv run src/lasr_stealth_evals/validation/generate_validation_set.py --output_dir validation --log_path $(LOG_PATH)

test:
	uv run pytest

eval-strategy-hint:
	make eval ARGS="--epochs 3 --turns 5 --model openrouter/openai/gpt-4.1 --strategy 1 -T agent_cot=True -T monitor_cot=False --max-connections 500"

eval-cot:
	$(INSPECT_EVAL_COMMON) -T agent_cot=False -T monitor_cot=False turns=5
	$(INSPECT_EVAL_COMMON) -T agent_cot=True -T monitor_cot=True turns=5

eval-cot-strats:
	$(eval MAX_CONNECTIONS := 500)
	$(INSPECT_EVAL_COMMON) -T agent_cot=False -T monitor_cot=False turns=5 generation_strategy=2
	$(INSPECT_EVAL_COMMON) -T agent_cot=True -T monitor_cot=True turns=5 generation_strategy=2

# Define the models to evaluate
MODELS = openrouter/openai/gpt-4.1 openrouter/anthropic/claude-3.7-sonnet openrouter/google/gemini-2.5-pro-preview-03-25 openrouter/deepseek/deepseek-chat

# Run evaluation for each model
eval-agent:
	@for model in $(MODELS); do \
		echo "Running evaluation for model: $$model"; \
		make eval ARGS="--epochs 5 --turns 5 --model $$model --monitor-model openrouter/openai/gpt-4.1 -T agent_cot=True -T monitor_cot=False --max-connections 300 --strategy 3"; \
		echo "Completed evaluation for model: $$model"; \
		echo "----------------------------------------"; \
	done
