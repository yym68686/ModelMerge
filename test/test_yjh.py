import os
from datetime import datetime

from aient.models import chatgpt
from aient.utils import prompt

API = os.environ.get('API', None)
API_URL = os.environ.get('API_URL', None)
GPT_ENGINE = os.environ.get('GPT_ENGINE', 'gpt-4o')
LANGUAGE = os.environ.get('LANGUAGE', 'Simplified Chinese')

current_date = datetime.now()
Current_Date = current_date.strftime("%Y-%m-%d")

systemprompt = os.environ.get('SYSTEMPROMPT', prompt.system_prompt.format(LANGUAGE, Current_Date))

bot = chatgpt(api_key=API, api_url=API_URL, engine=GPT_ENGINE, system_prompt=systemprompt)
for text in bot.ask_stream("arXiv:2210.10716 这篇文章讲了啥"):
# for text in bot.ask_stream("今天的微博热搜有哪些？"):
# for text in bot.ask_stream("你现在是什么版本？"):
    print(text, end="")