from inspect_ai.model import ChatMessageSystem, ChatMessage, ChatMessageUser
from typing import List
from lasr_stealth_evals.collusion.generate_samples import Metadata
from lasr_stealth_evals.collusion.config import METADATA_TAG


def update_system_message(metadata: Metadata, agent_type: str) -> ChatMessageSystem:
    """Get system message from sample metadata."""

    return ChatMessageSystem(content=getattr(metadata, agent_type))


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
