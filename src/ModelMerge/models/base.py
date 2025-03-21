import os
import httpx
import requests
from pathlib import Path
from collections import defaultdict

from ..utils import prompt
from ..core.utils import BaseAPI

class BaseLLM:
    def __init__(
        self,
        api_key: str = None,
        engine: str = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo",
        api_url: str = (os.environ.get("API_URL", None) or "https://api.openai.com/v1/chat/completions"),
        system_prompt: str = prompt.chatgpt_system_prompt,
        proxy: str = None,
        timeout: float = 600,
        max_tokens: int = None,
        temperature: float = 0.5,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        reply_count: int = 1,
        truncate_limit: int = None,
        use_plugins: bool = True,
        print_log: bool = False,
    ) -> None:
        self.api_key: str = api_key
        self.engine: str = engine
        self.api_url: str = BaseAPI(api_url or "https://api.openai.com/v1/chat/completions")
        self.system_prompt: str = system_prompt
        self.max_tokens: int = max_tokens
        self.truncate_limit: int = truncate_limit
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.presence_penalty: float = presence_penalty
        self.frequency_penalty: float = frequency_penalty
        self.reply_count: int = reply_count
        self.max_tokens: int = max_tokens or (
            4096
            if "gpt-4-1106-preview" in engine or "gpt-4-0125-preview" in engine or "gpt-4-turbo" in engine or "gpt-3.5-turbo-1106" in engine or "claude" in engine or "gpt-4o" in engine
            else 31000
            if "gpt-4-32k" in engine
            else 7000
            if "gpt-4" in engine
            else 16385
            if "gpt-3.5-turbo-16k" in engine
            # else 99000
            # if "claude-2.1" in engine
            else 4000
        )
        self.truncate_limit: int = truncate_limit or (
            127500
            if "gpt-4-1106-preview" in engine or "gpt-4-0125-preview" in engine or "gpt-4-turbo" in engine or "gpt-4o" in engine
            else 30500
            if "gpt-4-32k" in engine
            else 6500
            if "gpt-4" in engine
            else 14500
            if "gpt-3.5-turbo-16k" in engine or "gpt-3.5-turbo-1106" in engine
            else 98500
            if "claude-2.1" in engine
            else 3500
        )
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
        self.truncate_limit: int = 100000
        self.tokens_usage = defaultdict(int)
        self.function_calls_counter = {}
        self.function_call_max_loop = 10
        self.use_plugins = use_plugins
        self.print_log: bool = print_log

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
        model: str = "",
        pass_history: int = 9999,
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
        model: str = "",
        pass_history: int = 9999,
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
        model: str = "",
        pass_history: int = 9999,
        function_name: str = "",
        **kwargs,
    ):
        """
        Ask a question
        """
        pass

    async def ask_stream_async(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
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
        model: str = "",
        pass_history: int = 9999,
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        response = ""
        async for chunk in self.ask_stream_async(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            model=model or self.engine,
            pass_history=pass_history,
            **kwargs,
        ):
            response += chunk
        # full_response: str = "".join([r async for r in response])
        full_response: str = "".join(response)
        return full_response

    def ask(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 0,
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        response = self.ask_stream(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            model=model or self.engine,
            pass_history=pass_history,
            **kwargs,
        )
        full_response: str = "".join(response)
        return full_response

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