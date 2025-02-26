import numpy as np
from dotenv import load_dotenv
from prompts import prompts_dict
load_dotenv()
from langchain_openai import ChatOpenAI
import os
import time
import glob
import random



OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_LASR')

#chat = ChatOpenAI(model="gpt-3.5-turbo-0125")
llm = ChatOpenAI(model= 'gpt-3.5-turbo')

# create 10 customer support tickets

for i in range(10):
    try:
        ticket = llm.invoke(prompts_dict['create_tickets']).content
        with open("tickets/ticket_{}.txt".format(i), "w") as file:
            file.write(ticket)

    except Exception:
        print('Sleeping for {} sec'.format(10))
        time.sleep(10)

# Specify the folder containing the text files
folder_path = "tickets/"  

# Get a list of all .txt files in the folder
text_files = glob.glob(os.path.join(folder_path, "*.txt"))

# Loop through each file and append the new line
for file_path in text_files:
    with open(file_path, "a") as file:
        # add time needed to solve the tickets in each ticket
        number = random.randint(5, 45)
        new_line = "\n Time needed to solve ticket is {} minutes".format(number)  
        file.write(new_line)
        print(f"Appended new line to {file_path}")
