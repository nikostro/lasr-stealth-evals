


class Agent:

    def __init__(self):
        pass

    def read(self, read_prompt:str,
             ticket:str,
             protocol:str,
             warnings:dict,
             leaderboard:dict,
             action_history:dict,
             log:dict) -> str:
        """
        Reads the ticket and protocol and outputs decision
        Option A - Solve the ticket
        Option B - Pass to colleague
        Other - error happened and agent did not understand the task
        Log is used to record all actions and entire response + CoT

        Args:
            - reaad_prompt: agent prompt for reading the ticket
            - protocol - the protocol file that is inserted in the read_prompt
            - warnings - how many warnings agent already has, is inserted in the read_prompt
            - leaderboard - leaderboard added to the prompt
            - action_history - all actions agent took in the past
            - log- log for keeping track of all actions and agent output

        Return:
            - Option A or Option B or Error (agent fails to reply)
        """
        return 
    
    def solve(self, solve_prompt:str,
              ticket:str,
              protocol:str,
              log:dict) -> tuple[str, int]: 
        """
        Agent solves the ticket and output the list of tasks it finished according 
        to the protocol and also outputs the total effort points required for this task 
        Return: str - agent replies how they solved the ticket and int how much effort points used
        """
        return 

    def ask_for_help(self, ask_help_prompt:str,
                     ticket:str,
                     log:dict,
                     ):
        """
        Agent can ask for help and pass the ticket to colleague
        They can edit the ticket, add notes, make tweaks
        Return - return the name of colleague who you are passing the ticket 
        """
        return


        






    

