import os
import openai
from dotenv import load_dotenv

def load_env():
    load_dotenv("./resources/.env")
    return os.getenv("OPENAI_API_KEY")