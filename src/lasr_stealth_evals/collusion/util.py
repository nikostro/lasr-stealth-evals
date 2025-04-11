from inspect_ai.model import ChatMessageSystem, ChatMessage, ChatMessageUser
from typing import List
from lasr_stealth_evals.collusion.generate_samples import Config, Prompts
from lasr_stealth_evals.collusion.config import METADATA_TAG


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
