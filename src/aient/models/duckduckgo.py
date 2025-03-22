from types import TracebackType
from collections import defaultdict

import json
import httpx
from fake_useragent import UserAgent

class DuckChatException(httpx.HTTPError):
    """Base exception class for duck_chat."""


class RatelimitException(DuckChatException):
    """Raised for rate limit exceeded errors during API requests."""


class ConversationLimitException(DuckChatException):
    """Raised for conversation limit during API requests to AI endpoint."""


from enum import Enum
class ModelType(Enum):
    claude = "claude-3-haiku-20240307"
    llama = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    gpt4omini = "gpt-4o-mini"
    mixtral = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            # 对于完全匹配的情况
            for member in cls:
                if member.value == value:
                    return member

            # 对于部分匹配的情况
            for member in cls:
                if value in member.value:
                    return member

        return None

    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"ModelType({self.value!r})"

class Role(Enum):
    user = "user"
    assistant = "assistant"

import msgspec
class Message(msgspec.Struct):
    role: Role
    content: str

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key):
        return getattr(self, key)

class History(msgspec.Struct):
    model: ModelType
    messages: list[Message]

    def add_to_conversation(self, role: Role, message: str) -> None:
        self.messages.append(Message(role, message))

    def set_model(self, model_name: str) -> None:
        self.model = ModelType(model_name)

    def __getitem__(self, index: int) -> list[Message]:
        return self.messages[index]

    def __len__(self) -> int:
        return len(self.messages)

class UserHistory(msgspec.Struct):
    user_history: dict[str, History] = msgspec.field(default_factory=dict)

    def add_to_conversation(self, role: Role, message: str, convo_id: str = "default") -> None:
        if convo_id not in self.user_history:
            self.user_history[convo_id] = History(model=ModelType.claude, messages=[])
        self.user_history[convo_id].add_to_conversation(role, message)

    def get_history(self, convo_id: str = "default") -> History:
        if convo_id not in self.user_history:
            self.user_history[convo_id] = History(model=ModelType.claude, messages=[])
        return self.user_history[convo_id]

    def set_model(self, model_name: str, convo_id: str = "default") -> None:
        self.get_history(convo_id).set_model(model_name)

    def reset(self, convo_id: str = "default") -> None:
        self.user_history[convo_id] = History(model=ModelType.claude, messages=[])

    def get_all_convo_ids(self) -> list[str]:
        return list(self.user_history.keys())

    # 新增方法
    def __getitem__(self, convo_id: str) -> History:
        return self.get_history(convo_id)

class DuckChat:
    def __init__(
        self,
        model: ModelType = ModelType.claude,
        client: httpx.AsyncClient | None = None,
        user_agent: UserAgent | str = UserAgent(min_version=120.0),
    ) -> None:
        if isinstance(user_agent, str):
            self.user_agent = user_agent
        else:
            self.user_agent = user_agent.random

        self._client = client or httpx.AsyncClient(
            headers={
                "Host": "duckduckgo.com",
                "Accept": "text/event-stream",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://duckduckgo.com/",
                "User-Agent": self.user_agent,
                "DNT": "1",
                "Sec-GPC": "1",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "TE": "trailers",
            }
        )
        self.vqd: list[str | None] = []
        self.history = History(model, [])
        self.conversation = UserHistory({"default": self.history})
        self.__encoder = msgspec.json.Encoder()
        self.__decoder = msgspec.json.Decoder()

        self.tokens_usage = defaultdict(int)

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        await self._client.aclose()

    async def add_to_conversation(self, role: Role, message: Message, convo_id: str = "default") -> None:
        self.conversation.add_to_conversation(role, message, convo_id)

    async def get_vqd(self) -> None:
        """Get new x-vqd-4 token"""
        response = await self._client.get(
            "https://duckduckgo.com/duckchat/v1/status", headers={"x-vqd-accept": "1"}
        )
        if response.status_code == 429:
            try:
                err_message = self.__decoder.decode(response.content).get("type", "")
            except Exception:
                raise DuckChatException(response.text)
            else:
                raise RatelimitException(err_message)
        self.vqd.append(response.headers.get("x-vqd-4"))
        if not self.vqd:
            raise DuckChatException("No x-vqd-4")

    async def process_sse_stream(self, convo_id: str = "default"):
        # print("self.conversation[convo_id]", self.conversation[convo_id])
        async with self._client.stream(
            "POST",
            "https://duckduckgo.com/duckchat/v1/chat",
            headers={
                "Content-Type": "application/json",
                "x-vqd-4": self.vqd[-1],
            },
            content=self.__encoder.encode(self.conversation[convo_id]),
        ) as response:
            if response.status_code == 400:
                content = await response.aread()
                print("response.status_code", response.status_code, content)
            if response.status_code == 429:
                raise RatelimitException("Rate limit exceeded")

            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    yield line

    async def ask_stream_async(self, query, convo_id, model, **kwargs):
        """Get answer from chat AI"""
        if not self.vqd:
            await self.get_vqd()
        await self.add_to_conversation(Role.user, query, convo_id)
        self.conversation.set_model(model, convo_id)
        full_response = ""
        async for sse in self.process_sse_stream(convo_id):
            data = sse.lstrip("data: ")
            if data == "[DONE]":
                break
            resp: dict = json.loads(data)
            mess = resp.get("message")
            if mess:
                yield mess
                full_response += mess
        # await self.add_to_conversation(Role.assistant, full_response, convo_id)

    async def reset(self, convo_id: str = "default") -> None:
        self.conversation.reset(convo_id)

    # async def reask_question(self, num: int) -> str:
    #     """Get answer from chat AI"""

    #     if num >= len(self.vqd):
    #         num = len(self.vqd) - 1
    #     self.vqd = self.vqd[:num]

    #     if not self.history.messages:
    #         return ""

    #     if not self.vqd:
    #         await self.get_vqd()
    #         self.history.messages = [self.history.messages[0]]
    #     else:
    #         num = min(num, len(self.vqd))
    #         self.history.messages = self.history.messages[: (num * 2 - 1)]
    #     message = await self.get_answer()
    #     self.add_to_conversation(Role.assistant, message)

    #     return message
