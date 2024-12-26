import os
import openai

from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv(override=True)
from llama_index.core import Settings


# Now you can safely use environment variables, for example:
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_MODEL = os.getenv('OPENAI_API_MODEL')
openai.api_base = os.getenv('OPENAI_API_BASE')
Settings.base_url = os.getenv('OPENAI_API_BASE')
