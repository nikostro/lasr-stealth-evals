from langchain_openai import ChatOpenAI
from prompts.agent_prompts import prompts_dict
import json

llm = ChatOpenAI(model= 'gpt-3.5-turbo')

def get_file(path:str):
    with open(path, "r") as file:
        ticket = file.read()
    return ticket


def get_effort_points(content:str):
    prompt = prompts_dict['utils_parse_effort_points'].format(content=content)
    reply = llm.invoke(prompt).content
    try:
        return int(reply)
    except ValueError:
        return "No effort points found"


def get_agent_name_to_who_ticket_passed (content:str):
    prompt = prompts_dict['utils_get_agent_name'].format(content=content)
    reply = llm.invoke(prompt).content
    return reply

def get_modify_or_not (content:str):
    prompt = prompts_dict['utils_get_modify_or_not'].format(content=content)
    reply = llm.invoke(prompt).content
    return reply

def get_tickets_list(path:str):
    with open(path, "r") as file:
        data = json.load(file)
    tickets = []
    for i in range(len(data)):
        tickets.append(data[i]['description'])
    return tickets

def load_json(path:str):
    with open(path, "r") as file:
        data = json.load(file)
    return data

def save_json(path:str, data):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

