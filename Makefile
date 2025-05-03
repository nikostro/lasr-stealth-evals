.PHONY: eval validation

# Default configuration
FAIL_ON_ERROR ?= 0.05

# Common evaluation parameters
INSPECT_EVAL_BASE = uv run inspect eval src/lasr_stealth_evals/collusion/task.py
INSPECT_EVAL_COMMON = $(INSPECT_EVAL_BASE) --max-connections 400 --fail-on-error $(FAIL_ON_ERROR) --epochs 1 --model openrouter/google/gemini-2.5-flash-preview

# Small run for testing purposes
eval-test:
	$(INSPECT_EVAL_COMMON) --limit 30 -T agent_model=openrouter/google/gemini-2.5-flash-preview -T monitor_model=openrouter/google/gemini-2.5-flash-preview -T turns=5 -T use_hint=False -T use_justification=True -T is_cot_monitored=True -T generation_strategy=0

eval-validation:
	$(INSPECT_EVAL_BASE) --model openrouter/openai/gpt-4.1 --max-connections 400 -T turns=3 -T use_hint=False -T use_justification=True -T is_cot_monitored=True -T generation_strategy=1 $(ARGS)

# Validation - Usage: make validation LOG_PATH=/path/to/your/log/file
validation:
	uv run src/lasr_stealth_evals/validation/generate_validation_set.py --output_dir data/validation --log_path $(LOG_PATH)

validation-eval:
	uv run inspect eval src/lasr_stealth_evals/validation/validation_eval.py --model "openrouter/openai/gpt-4.1" --log-dir logs/validation

test:
	uv run pytest

eval-strategy-hint:
	make eval ARGS="--epochs 3 --turns 5 --model openrouter/openai/gpt-4.1 --strategy 1 -T agent_cot=True -T monitor_cot=False --max-connections 500"

eval-cot:
	echo "Running COT evaluation..."
	echo "Generation Strategy: None"
	$(INSPECT_EVAL_COMMON) -T agent_cot=False -T monitor_cot=False -T turns=5 -T generation_strategy=3
	$(INSPECT_EVAL_COMMON) -T agent_cot=True -T monitor_cot=False -T turns=5 -T generation_strategy=3
	$(INSPECT_EVAL_COMMON) -T agent_cot=True -T monitor_cot=True -T turns=5 -T generation_strategy=3
	echo "Generation Strategy: All"
	$(INSPECT_EVAL_COMMON) -T agent_cot=False -T monitor_cot=False -T turns=5 -T generation_strategy=2
	$(INSPECT_EVAL_COMMON) -T agent_cot=True -T monitor_cot=False -T turns=5 -T generation_strategy=2
	$(INSPECT_EVAL_COMMON) -T agent_cot=True -T monitor_cot=True -T turns=5 -T generation_strategy=2

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
