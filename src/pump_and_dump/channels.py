class MessagingChannels:
    PUBLIC = "public_chat"

    @staticmethod
    def agent_dm(agent_a: str, agent_b: str) -> str:
        sorted_agents = sorted([agent_a, agent_b])
        return f"dm-{sorted_agents[0]}-{sorted_agents[1]}"

    @staticmethod
    def overseer_dm(agent_id: str) -> str:
        return f"dm-{agent_id}-overseer"
