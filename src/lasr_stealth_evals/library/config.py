import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
