import os
from aient.models import chatgpt

API = os.environ.get('API', None)
API_URL = os.environ.get('API_URL', None)
GPT_ENGINE = os.environ.get('GPT_ENGINE', 'gpt-4o')

systemprompt = (
    "你是一位旅行规划专家。你需要帮助用户规划旅行行程，给出合理的行程安排。"
    "- 如果用户提及要从一个城市前往另外一个城市，必须使用 get_Round_trip_flight_price 查询两个城市半年内往返机票价格信息。"
    "- 在规划行程之前，必须使用 get_city_tarvel_info 查询城市的景点旅行攻略信息。"
    "- 查询攻略后，你需要分析用户个性化需求。充分考虑用户的年龄，情侣，家庭，朋友，儿童，独自旅行等情况。排除不适合用户个性化需求的景点。之后输出符合用户需求的景点。"
    "- 综合用户游玩时间，适合用户个性化需求的旅游城市景点，机票信息和预算，给出真实准确的旅游行程，包括游玩时长、景点之间的交通方式和移动距离，每天都要给出总的游玩时间。"
    "- 根据查到的景点介绍结合你自己的知识，每个景点必须包含你推荐的理由和景点介绍。介绍景点用户游玩的景点，景点介绍尽量丰富精彩，吸引用户眼球，不要直接复述查到的景点介绍。"
    "- 每个景点都要标注游玩时间、景点之间的交通方式和移动距离还有生动的景点介绍"
    "- 尽量排满用户的行程，不要有太多空闲时间。"
)
bot = chatgpt(api_key=API, api_url=API_URL, engine=GPT_ENGINE, system_prompt=systemprompt)
for text in bot.ask_stream("我在上海想去重庆旅游，我只有2000元预算，我想在重庆玩一周，你能帮我规划一下吗？"):
# for text in bot.ask_stream("我在广州市，想周一去香港，周四早上回来，是去游玩，请你帮我规划整个行程。包括细节，如交通，住宿，餐饮，价格，等等，最好细节到每天各个部分的时间，花费，等等，尽量具体，用户一看就能直接执行的那种"):
# for text in bot.ask_stream("上海有哪些好玩的地方？"):
# for text in bot.ask_stream("just say test"):
# for text in bot.ask_stream("我在上海想去重庆旅游，我只有2000元预算，我想在重庆玩一周，你能帮我规划一下吗？"):
# for text in bot.ask_stream("我在上海想去重庆旅游，我有一天的时间。你能帮我规划一下吗？"):
    print(text, end="")