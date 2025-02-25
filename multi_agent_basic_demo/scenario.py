import glob
import os
from langchain_openai import ChatOpenAI
from prompts import prompts_dict
import re
from collections import deque


def parse_leaderboard(leaderboard):
    lb = ''
    for name, stats in leaderboard.items():
        lb += f"{name.title()} spent {stats['time']} minutes in total to solve {stats['tickets']} tikets\n"
    return lb

# gpt-3.5-turbo
# o1-preview
# gpt-4o
llm = ChatOpenAI(model= 'gpt-3.5-turbo')

# Specify the folder containing the text files
folder_path = "tickets/"  

# Get a list of all .txt files in the folder
text_files = glob.glob(os.path.join(folder_path, "*.txt"))

leaderboard = {
    "olivia": {
        "time": 0,
        "tickets": 0
    },
    "alex": {
        "time": 0,
        "tickets": 0
    },
    "ben": {
        "time": 0,
        "tickets": 0
    }
}


warnings = {
    "olivia": 0,
    "alex": 0,
    "ben": 0
}

agents_list = deque(['olivia', 'alex', 'ben'])

# run basic simulation (not complete - did not implement oversight mechanism)
while text_files:
    
    agent_name = agents_list.popleft()
    agents_list.append(agent_name)
    ticket_path = text_files.pop()
    #load the file

    with open(ticket_path, "r") as file:
        ticket = file.read()  # Read the entire content of the file
        print(ticket)

    ld = parse_leaderboard(leaderboard)

    if agent_name == 'olivia':
        prompt = prompts_dict['olivia_prompt'].format(ticket=ticket, 
                                                leaderboard=ld,
                                                warnings = warnings["olivia"])
    elif agent_name == 'alex':
        prompt = prompts_dict['general_agent_prompt'].format(ticket=ticket, 
                                                leaderboard=ld,
                                                warnings = warnings["alex"])
        
    else:
        prompt = prompts_dict['general_agent_prompt'].format(ticket=ticket, 
                                                leaderboard=ld,
                                                warnings = warnings["ben"])

    reply = llm.invoke(prompt).content

    print(reply)

    # write to log
    with open('log.txt', "a") as file:
        # add time needed to solve the tickets in each ticket
        new_line = reply + "\n"
        file.write(agent_name + "\n")
        file.write(new_line)
        file.write("------------------------- \n")
        print("Added to log")

    # which action did Agent chose?

    if "Option A" in reply:
        match = re.search(r'(\d+)\s+minutes', ticket)
        if match:
            minutes = int(match.group(1))
            leaderboard[agent_name]["time"] += minutes
            leaderboard[agent_name]["tickets"] += 1
            print("Extracted number:", minutes)
        else:
            print("No match found.")

    else:
        # pass it to another person
        print("Chose option B")

        
