import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Now you can safely use environment variables, for example:
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_MODEL = os.getenv('OPENAI_API_MODEL')
