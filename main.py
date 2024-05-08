import os
from datetime import datetime

from utils import prompt
from models import chatgpt

LANGUAGE = os.environ.get('LANGUAGE', 'Simplified Chinese')
GPT_ENGINE = os.environ.get('GPT_ENGINE', 'gpt-4-turbo-2024-04-09')
API = os.environ.get('API', None)
current_date = datetime.now()
Current_Date = current_date.strftime("%Y-%m-%d")

systemprompt = os.environ.get('SYSTEMPROMPT', prompt.system_prompt.format(LANGUAGE, Current_Date))
bot = chatgpt(api_key=f"{API}", engine=GPT_ENGINE, system_prompt=systemprompt)
for text in bot.ask_stream("hi"):
    print(text, end="")