from inspect_ai.model import ChatMessageSystem, ChatMessage, ChatMessageUser
from typing import List
from lasr_stealth_evals.collusion.generate_samples import Config, Prompts
from lasr_stealth_evals.collusion.config import METADATA_TAG


def update_system_message(sample_metadata: Config, agent_type: str) -> ChatMessageSystem:
    """Update system message based on sample metadata."""

    # Map agent types to their corresponding secret fields
    secret_mapping = {"agent": "strategies", "monitor": "hints"}
    print(sample_metadata.secret)
    assert hasattr(sample_metadata.secret, secret_mapping[agent_type])
    secret = getattr(sample_metadata.secret, secret_mapping[agent_type])
    prompts: Prompts = getattr(sample_metadata, agent_type)

    instructions = " ".join(
        (
            prompts.persona,
            prompts.task,
            prompts.monitor,
            "\n".join([f"{s.description}" for s in secret]),  # hint or strategy
            prompts.reply,
        )
    )

    return ChatMessageSystem(content=instructions)


def filter_first_user_message(messages: List[ChatMessage], metadata_tag: str = METADATA_TAG) -> List[ChatMessage]:
    """Remove user messages that contain the metadata tag from the chat history.

    Args:
       messages: Messages to filter.
       metadata_tag: Tag to identify metadata messages.

    Returns:
       Messages without metadata-tagged user messages.
    """
    return [
        message
        for message in messages
        if not (isinstance(message, ChatMessageUser) and metadata_tag in message.content)
    ]
