import numpy
from langchain_openai import ChatOpenAI


class Agent:

    def __init__(self, model:str):
        self.model = model
        self.llm = ChatOpenAI(model=self.model)


    def read(self, prompt_read:str,
             ticket:str,
             protocol:str,
             warnings:dict,
             effort_points:int,
             leaderboard:dict,
             action_history:dict) -> str:
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
        #llm = ChatOpenAI(model=self.model)
        prompt = prompt_read.format(
            protocol=protocol,
            leaderboard = leaderboard,
            warnings=warnings,
            effort_points=str(effort_points),
            action_history = action_history,
            ticket=ticket)
        reply = self.llm.invoke(prompt).content
        #print('=============================================')
        #print(prompt)
        #print('=============================================')
        return reply, prompt
    

    def solve(self, prompt_solve:str,
              ticket:str,
              protocol:str,
              warnings:dict,
              effort_points:int,
              action_history:dict,
              leaderboard:dict) -> str: 
        
        """
        Agent solves the ticket and output the list of tasks it finished according 
        to the protocol and also outputs the total effort points required for this task 
        Return: str - agent replies how they solved the ticket and int how much effort points used
        """
        prompt = prompt_solve.format(
            protocol=protocol,
            leaderboard = leaderboard,
            warnings=warnings,
            effort_points=str(effort_points),
            action_history = action_history,
            ticket=ticket)
        reply = self.llm.invoke(prompt).content
        return reply, prompt


    def ask_for_help(self, ask_help_prompt:str,
                        ticket:str,
                        protocol:str,
                        warnings:dict,
                        effort_points:int,
                        action_history:dict,
                        leaderboard:dict) -> str: 
        """
        Agent can ask for help and pass the ticket to colleague
        They can edit the ticket, add notes, make tweaks
        Return - return the name of colleague who you are passing the ticket 
        """
        prompt = ask_help_prompt.format(
                protocol=protocol,
                leaderboard = leaderboard,
                warnings=warnings,
                effort_points=str(effort_points),
                action_history = action_history,
                ticket=ticket)
        reply = self.llm.invoke(prompt).content
        return reply, prompt

    
    def modify_ticket(self, modify_ticket_prompt:str,
                        ticket:str,
                        protocol:str,
                        warnings:dict,
                        effort_points:int,
                        action_history:dict,
                        colleague_name:str,
                        leaderboard:dict) -> str: 
        """
        Agent can ask for help and pass the ticket to colleague
        They can edit the ticket, add notes, make tweaks
        Return - return the name of colleague who you are passing the ticket 
        """
        prompt = modify_ticket_prompt.format(
                protocol=protocol,
                leaderboard = leaderboard,
                warnings=warnings,
                effort_points=str(effort_points),
                action_history = action_history,
                colleague_name=colleague_name,
                ticket=ticket)
        reply = self.llm.invoke(prompt).content
        return reply, prompt



        






    