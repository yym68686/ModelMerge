import os
from datetime import datetime

from aient.utils import prompt
from aient.models import chatgpt, claude3, gemini, groq
LANGUAGE = os.environ.get('LANGUAGE', 'Simplified Chinese')
GPT_ENGINE = os.environ.get('GPT_ENGINE', 'gpt-4-turbo-2024-04-09')

API = os.environ.get('API', None)
API_URL = os.environ.get('API_URL', None)

CLAUDE_API = os.environ.get('CLAUDE_API', None)
GOOGLE_AI_API_KEY = os.environ.get('GOOGLE_AI_API_KEY', None)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', None)

current_date = datetime.now()
Current_Date = current_date.strftime("%Y-%m-%d")

message = "https://arxiv.org/abs/2404.02041 这篇论文讲了啥？"
systemprompt = os.environ.get('SYSTEMPROMPT', prompt.chatgpt_system_prompt)
# systemprompt = os.environ.get('SYSTEMPROMPT', prompt.system_prompt.format(LANGUAGE, Current_Date))
# systemprompt = (
#     "你是一位旅行专家。你可以规划旅行行程，如果用户有预算限制，还需要查询机票价格。结合用户的出行时间，给出合理的行程安排。"
#     "在规划行程之前，必须先查找旅行攻略搜索景点信息，即使用 get_city_tarvel_info 查询景点信息。查询攻略后，你需要分析用户个性化需求给出合理的行程安排。充分考虑用户的年龄，情侣，家庭，朋友，儿童，独自旅行等情况。"
#     "你需要根据用户给出的地点和预算，给出真实准确的行程，包括游玩时长、景点之间的交通方式和移动距离，每天都要给出总的游玩时间。"
#     "给用户介绍景点的时候，根据查到的景点介绍结合你自己的知识，景点介绍尽量丰富精彩，吸引用户眼球，不要直接复述查到的景点介绍。"
#     "尽量排满用户的行程，不要有太多空闲时间。"
#     "你还可以根据用户的需求，给出一些旅行建议。"
# )
bot = chatgpt(api_key=API, api_url=API_URL , engine=GPT_ENGINE, system_prompt=systemprompt)
# bot = claude3(api_key=CLAUDE_API, engine=GPT_ENGINE, system_prompt=systemprompt)
# bot = gemini(api_key=GOOGLE_AI_API_KEY, engine=GPT_ENGINE, system_prompt=systemprompt)
# bot = groq(api_key=GROQ_API_KEY, engine=GPT_ENGINE, system_prompt=systemprompt)
for text in bot.ask_stream(message):
# for text in bot.ask_stream("今天的微博热搜有哪些？"):
# for text in bot.ask_stream("250m usd = cny"):
# for text in bot.ask_stream("我在广州市，想周一去香港，周四早上回来，是去游玩，请你帮我规划整个行程。包括细节，如交通，住宿，餐饮，价格，等等，最好细节到每天各个部分的时间，花费，等等，尽量具体，用户一看就能直接执行的那种"):
# for text in bot.ask_stream("英伟达最早支持杜比视界的显卡是哪一代"):
# for text in bot.ask_stream("100个斐波纳切数列的和是多少"):
# for text in bot.ask_stream("上海有哪些好玩的地方？"):
# for text in bot.ask_stream("https://arxiv.org/abs/2404.02041 这篇论文讲了啥？"):
# for text in bot.ask_stream("今天伊朗总统目前的情况怎么样？"):
# for text in bot.ask_stream("我不是很懂y[..., 2]，y[..., 2] - y[:, 0:1, 0:1, 2]，y[:, 0:1, 0:1, 2]这些对张量的slice操作，给我一些练习demo代码，专门给我巩固这些张量复杂操作。让我从易到难理解透彻所有这样类型的张量操作。"):
# for text in bot.ask_stream("just say test"):
# for text in bot.ask_stream("画一只猫猫"):
# for text in bot.ask_stream("我在上海想去重庆旅游，我只有2000元预算，我想在重庆玩一周，你能帮我规划一下吗？"):
# for text in bot.ask_stream("我在上海想去重庆旅游，我有一天的时间。你能帮我规划一下吗？"):
    print(text, end="")

# print("\n bot tokens usage", bot.tokens_usage)