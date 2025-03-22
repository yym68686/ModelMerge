import os
from aient.models import chatgpt

API = os.environ.get('API', None)
API_URL = os.environ.get('API_URL', None)
GPT_ENGINE = os.environ.get('GPT_ENGINE', 'gpt-4o')

systemprompt = (
    "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally"
)
bot = chatgpt(api_key=API, api_url=API_URL, engine=GPT_ENGINE, system_prompt=systemprompt, print_log=True)
for text in bot.ask_stream("搜索上海的天气"):
# for text in bot.ask_stream("我在广州市，想周一去香港，周四早上回来，是去游玩，请你帮我规划整个行程。包括细节，如交通，住宿，餐饮，价格，等等，最好细节到每天各个部分的时间，花费，等等，尽量具体，用户一看就能直接执行的那种"):
# for text in bot.ask_stream("上海有哪些好玩的地方？"):
# for text in bot.ask_stream("just say test"):
# for text in bot.ask_stream("我在上海想去重庆旅游，我只有2000元预算，我想在重庆玩一周，你能帮我规划一下吗？"):
# for text in bot.ask_stream("我在上海想去重庆旅游，我有一天的时间。你能帮我规划一下吗？"):
    print(text, end="")