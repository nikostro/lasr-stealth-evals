
import numpy as np
from dotenv import load_dotenv
from prompts import prompts_dict
load_dotenv()
from langchain_openai import ChatOpenAI
import os
import time
import glob
import random
import random
import datetime
import json

word = "Adidas"

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_LASR')

#chat = ChatOpenAI(model="gpt-3.5-turbo-0125")
llm = ChatOpenAI(model= 'gpt-3.5-turbo')

def random_time_generator(start="08:00", end="16:00"):
    # Define the start and end times
    start_time = datetime.datetime.strptime(start, "%H:%M")
    end_time = datetime.datetime.strptime(end, "%H:%M")
    # Generate a random number of seconds between the start and end times
    random_seconds = random.randint(0, int((end_time - start_time).total_seconds()))

    # Compute the random time
    random_time = start_time + datetime.timedelta(seconds=random_seconds)
    # Print the random time in HH:MM format
    return random_time.strftime("%I:%M %p")

# create email thread
for i in range(10):
    try:
        email_thread = llm.invoke(prompts_dict['generate_email_threads']).content
        with open("data/emails/email_thread_{}.txt".format(i), "w") as file:
            file.write(email_thread)

    except Exception:
        print('Sleeping for {} sec'.format(10))
        time.sleep(10)


# create the final prompt data
folder_path = "data/emails/"  
text_files = glob.glob(os.path.join(folder_path, "*.txt")) #get the list
count = 0
for file_path in text_files:
    rand_time = random_time_generator(start="08:00", end="16:00")
    count += 1

    with open(file_path, "r") as file:
        content = file.read()

    prompt_json = {}

    prompt = prompts_dict["draft_reply"].format(
    time = rand_time,
    email_thread = content,
    word = word)
        
    prompt_json["input"] = [{"role":"user", "content":prompt}]
    prompt_json["time"] = rand_time
    prompt_json["target"] = word

    # Write the data to a JSON file
    with open("data/prompts/data_{}.json".format(count), "w", encoding="utf-8") as json_file:
        json.dump(prompt_json, json_file, indent=4)
    

folder_path = "data/prompts/"  
text_files = glob.glob(os.path.join(folder_path, "*.json")) #get the list

data_list = []

for file_path in text_files:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    data_list.append(data)
    

# Save the updated list as a new JSON file
with open("data/data.json", "w", encoding="utf-8") as file:
    json.dump(data_list, file, indent=4)



