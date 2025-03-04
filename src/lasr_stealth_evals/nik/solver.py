from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver


@solver
def add_context(user_message: ChatMessageUser, assistant_message: ChatMessageAssistant):
    """This eval requires info about a first email. This provides the context needed for the model to evade oversight."""

    async def solve(state: TaskState, generate: Generate):
        # Find the last system message index
        last_system_index = -1
        for i in reversed(range(len(state.messages))):
            if isinstance(state.messages[i], ChatMessageSystem):
                last_system_index = i
                break

        # Insert messages after the last system message
        state.messages[last_system_index + 1 : last_system_index + 1] = [user_message, assistant_message]
        return state

    return solve
