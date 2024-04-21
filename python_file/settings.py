import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(verbose=True)

dotenv_path = Path.cwd().parent / 'env_file' / '.env.work'
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    print("No .env file found in the project root directory")
    exit(1)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
JAPANESE_FLG = os.environ.get("JAPANESE_FLG")
MAKE_BING_QUERY_TIMEOUT = os.environ.get("MAKE_BING_QUERY_TIMEOUT")
GET_BING_RESULT_TIMEOUT = os.environ.get("GET_BING_RESULT_TIMEOUT")
REVISE_ANSWER_TIMEOUT = os.environ.get("REVISE_ANSWER_TIMEOUT")