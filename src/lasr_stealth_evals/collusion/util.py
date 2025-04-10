from inspect_ai.model import ChatMessageSystem
from lasr_stealth_evals.collusion.generate_samples import Config, Prompts


def update_system_message(sample_metadata: Config, system_message: ChatMessageSystem) -> ChatMessageSystem:
    """Update system message based on sample metadata."""
    content = system_message.content
    assert type(content) is str

    content_list = content.split("\n")
    agent_type = content_list[0].strip()

    # Map agent types to their corresponding secret fields
    secret_mapping = {"agent": "strategy_description", "monitor": "hint"}
    assert hasattr(sample_metadata.secret, secret_mapping[agent_type])
    secret = getattr(sample_metadata.secret, secret_mapping[agent_type])
    prompts: Prompts = getattr(sample_metadata, agent_type)

    instructions = " ".join(
        (
            prompts.persona,
            prompts.task,
            secret,  # hint or strategy
            prompts.reply,
        )
    )

    content = "\n".join([instructions] + content_list[1:])
    return system_message.model_copy(update={"content": content})
