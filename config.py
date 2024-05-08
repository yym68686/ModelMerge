import os
from dotenv import load_dotenv
load_dotenv()
import utils.prompt as prompt

WEB_HOOK = os.environ.get('WEB_HOOK', None)
BOT_TOKEN = os.environ.get('BOT_TOKEN', None)
PORT = int(os.environ.get('PORT', '8080'))
NICK = os.environ.get('NICK', None)
API = os.environ.get('API', None)
PASS_HISTORY = (os.environ.get('PASS_HISTORY', "False") == "False") == False
USE_GOOGLE = (os.environ.get('USE_GOOGLE', "True") == "False") == False
if os.environ.get('GOOGLE_API_KEY', None) == None and os.environ.get('GOOGLE_CSE_ID', None) == None:
    USE_GOOGLE = False
temperature = float(os.environ.get('temperature', '0.5'))
GPT_ENGINE = os.environ.get('GPT_ENGINE', 'gpt-4-turbo-2024-04-09')
# DEFAULT_SEARCH_MODEL = os.environ.get('DEFAULT_SEARCH_MODEL', 'gpt-3.5-turbo-1106') gpt-3.5-turbo-16k
API_URL = os.environ.get('API_URL', 'https://api.openai.com/v1/chat/completions')
# PDF_EMBEDDING = (os.environ.get('PDF_EMBEDDING', "True") == "False") == False
LANGUAGE = os.environ.get('LANGUAGE', 'Simplified Chinese')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', None)
GOOGLE_AI_API_KEY = os.environ.get('GOOGLE_AI_API_KEY', None)
CUSTOM_MODELS = os.environ.get('CUSTOM_MODELS', None)
if CUSTOM_MODELS:
    CUSTOM_MODELS_LIST = [id for id in CUSTOM_MODELS.split(",")]
else:
    CUSTOM_MODELS_LIST = None


from datetime import datetime
current_date = datetime.now()
Current_Date = current_date.strftime("%Y-%m-%d")
systemprompt = os.environ.get('SYSTEMPROMPT', prompt.system_prompt.format(LANGUAGE, Current_Date))
claude_systemprompt = os.environ.get('SYSTEMPROMPT', prompt.claude_system_prompt)

# from utils.chatgpt2api import Chatbot as GPT
# from utils.chatgpt2api import Imagebot, claudebot, groqbot, claude3bot, gemini_bot
# if API:
#     ChatGPTbot = GPT(api_key=f"{API}", engine=GPT_ENGINE, system_prompt=systemprompt, temperature=temperature)

#     translate_bot = GPT(api_key=f"{API}", engine=GPT_ENGINE, system_prompt=systemprompt, temperature=temperature)
#     copilot_bot = GPT(api_key=f"{API}", engine=GPT_ENGINE, system_prompt=prompt.search_system_prompt.format(LANGUAGE), temperature=temperature)
#     dallbot = Imagebot(api_key=f"{API}")
# else:
#     ChatGPTbot = None

# ClaudeAPI = os.environ.get('claude_api_key', None)
# if ClaudeAPI:
#     claudeBot = claudebot(api_key=f"{ClaudeAPI}", system_prompt=claude_systemprompt)
#     claude3Bot = claude3bot(api_key=f"{ClaudeAPI}", system_prompt=claude_systemprompt)

# if GROQ_API_KEY:
#     groqBot = groqbot(api_key=f"{GROQ_API_KEY}")
# if GOOGLE_AI_API_KEY:
#     gemini_Bot = gemini_bot(api_key=f"{GOOGLE_AI_API_KEY}")

whitelist = os.environ.get('whitelist', None)
if whitelist:
    whitelist = [int(id) for id in whitelist.split(",")]
ADMIN_LIST = os.environ.get('ADMIN_LIST', None)
if ADMIN_LIST:
    ADMIN_LIST = [int(id) for id in ADMIN_LIST.split(",")]
GROUP_LIST = os.environ.get('GROUP_LIST', None)
if GROUP_LIST:
    GROUP_LIST = [int(id) for id in GROUP_LIST.split(",")]

PLUGINS = {
    "SEARCH_USE_GPT": (os.environ.get('SEARCH_USE_GPT', "True") == "False") == False,
    # "USE_G4F": (os.environ.get('USE_G4F', "False") == "False") == False,
    "DATE": True,
    "URL": True,
    "VERSION": True,
}

class userConfig:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.language = LANGUAGE
        self.temperature = temperature
        self.engine = GPT_ENGINE
        self.system_prompt = systemprompt
        self.search_system_prompt = prompt.search_system_prompt.format(self.language)
        self.search_model = "gpt-3.5-turbo-1106"



def get_plugins_status(item):
    return "✅" if PLUGINS[item] else "☑️"

def delete_model_digit_tail(lst):
    for i in range(len(lst) - 1, -1, -1):
        if not lst[i].isdigit():
            if i == len(lst) - 1:
                return "-".join(lst)
            else:
                return "-".join(lst[:i + 1])

initial_model = [
    "gpt-4-turbo-2024-04-09",
    "gpt-3.5-turbo",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

if GROQ_API_KEY:
    initial_model.extend([
        "mixtral-8x7b-32768",
        "llama3-70b-8192",
    ])
if GOOGLE_AI_API_KEY:
    initial_model.extend([
        "gemini-1.5-pro-latest",
    ])

if CUSTOM_MODELS_LIST:
    delete_models = [model[1:] for model in CUSTOM_MODELS_LIST if model[0] == "-"]
    for target in delete_models:
        for model in initial_model:
            if target in model:
                initial_model.remove(model)

    initial_model.extend([model for model in CUSTOM_MODELS_LIST if model not in initial_model and model[0] != "-"])