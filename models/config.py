import os
import httpx
import requests
from pathlib import Path

from utils import prompt



# if config.CUSTOM_MODELS_LIST:
#     ENGINES.extend(config.CUSTOM_MODELS_LIST)

ENGINES = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-turbo-2024-04-09",
    "mixtral-8x7b-32768",
    "llama2-70b-4096",
    "llama3-70b-8192",
    "claude-2.1",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
    "gemini-1.5-pro-latest",
]

CUSTOM_MODELS = os.environ.get('CUSTOM_MODELS', None)
if CUSTOM_MODELS:
    CUSTOM_MODELS_LIST = [id for id in CUSTOM_MODELS.split(",")]
else:
    CUSTOM_MODELS_LIST = None

PLUGINS = {
    "SEARCH_USE_GPT": (os.environ.get('SEARCH_USE_GPT', "True") == "False") == False,
    # "USE_G4F": (os.environ.get('USE_G4F', "False") == "False") == False,
    "DATE": True,
    "URL": True,
    "VERSION": True,
}

LANGUAGE = os.environ.get('LANGUAGE', 'Simplified Chinese')


class BaseLLM:
    def __init__(
        self,
        api_key: str,
        engine: str = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo",
        proxy: str = None,
        timeout: float = 600,
        max_tokens: int = None,
        temperature: float = 0.5,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        reply_count: int = 1,
        truncate_limit: int = None,
        system_prompt: str = prompt.chatgpt_system_prompt,
    ) -> None:
        self.engine: str = engine
        self.api_key: str = api_key
        self.system_prompt: str = system_prompt
        self.max_tokens: int = max_tokens
        self.truncate_limit: int = truncate_limit
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.presence_penalty: float = presence_penalty
        self.frequency_penalty: float = frequency_penalty
        self.reply_count: int = reply_count
        self.timeout: float = timeout
        self.proxy = proxy
        self.session = requests.Session()
        self.session.proxies.update(
            {
                "http": proxy,
                "https": proxy,
            },
        )
        if proxy := (
            proxy or os.environ.get("all_proxy") or os.environ.get("ALL_PROXY") or None
        ):
            if "socks5h" not in proxy:
                self.aclient = httpx.AsyncClient(
                    follow_redirects=True,
                    proxies=proxy,
                    timeout=timeout,
                )
        else:
            self.aclient = httpx.AsyncClient(
                follow_redirects=True,
                proxies=proxy,
                timeout=timeout,
            )

        self.conversation: dict[str, list[dict]] = {
            "default": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
        }
        self.function_calls_counter = {}
        self.function_call_max_loop = 10

    def add_to_conversation(
        self,
        message: list,
        role: str,
        convo_id: str = "default",
        function_name: str = "",
    ) -> None:
        """
        Add a message to the conversation
        """
        pass

    def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        pass

    def truncate_conversation(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        **kwargs,
    ) -> None:
        """
        Truncate the conversation
        """
        pass

    def extract_values(self, obj):
        pass

    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        pass

    def get_message_token(self, url, json_post):
        pass


    def get_post_body(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        **kwargs,
    ):
        pass

    def get_max_tokens(self, convo_id: str) -> int:
        """
        Get max tokens
        """
        pass

    def ask_stream(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        function_name: str = "",
        **kwargs,
    ):
        """
        Ask a question
        """
        pass

    async def ask_async(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        pass

    def ask(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = None,
        pass_history: bool = True,
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        pass

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            {"role": "system", "content": system_prompt or self.system_prompt},
        ]

    def save(self, file: str, *keys: str) -> None:
        """
        Save the Chatbot configuration to a JSON file
        """
        pass

    def load(self, file: Path, *keys_: str) -> None:
        """
        Load the Chatbot configuration from a JSON file
        """
        pass