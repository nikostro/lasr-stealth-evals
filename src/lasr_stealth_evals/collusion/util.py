from typing import Any
from inspect_ai.model import ChatMessageSystem


def update_system_message(sample_metadata: dict[str, Any], system_message: ChatMessageSystem) -> ChatMessageSystem:
    """Update system message based on sample metadata."""
    content = system_message.content
    assert type(content) is str

    content_list = content.split("\n")
    agent_type = content_list[0].strip()
    assert agent_type in sample_metadata
    instructions = " ".join(sample_metadata[agent_type].values())
    content = "\n".join([instructions] + content_list[1:])
    return system_message.model_copy(update={"content": content})
