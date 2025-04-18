from typing import List

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser

from lasr_stealth_evals.collusion.config import METADATA_TAG
from lasr_stealth_evals.collusion.generate_samples import Metadata


class Role(ChatMessageAssistant):
    pass


def get_system_message(metadata: Metadata, agent_type: str) -> ChatMessageSystem:
    """Get system message from sample metadata."""

    return ChatMessageSystem(content=getattr(metadata, agent_type + "_prompt"))


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


def swap_roles(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Swap the roles of the messages in the chat history."""
    swapped_messages = []
    i = len(messages) - 1
    while i >= 0:
        message = messages[i]
        if isinstance(message, ChatMessageAssistant):
            swapped_messages.append(ChatMessageUser(content=message.content))
        elif isinstance(message, ChatMessageUser):
            swapped_messages.append(ChatMessageAssistant(content=message.content))
        elif isinstance(message, ChatMessageTool):
            swapped_messages.append(message)
            i -= 1
        else:
            swapped_messages.append(message)
        i -= 1
    return swapped_messages[::-1]


def add_role_message(messages: List[ChatMessage], agent_type: str) -> List[ChatMessage]:
    """Add a role message to the chat history."""
    messages.append(Role(content=f"You are {agent_type}."))
    return messages


def remove_role_message(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Remove the role message from the chat history."""
    return [message for message in messages if not isinstance(message, Role)]
