import os
import re
import json
import copy
import httpx
import asyncio
import requests
from typing import Set
from typing import Union
from pathlib import Path


from .base import BaseLLM
from ..plugins import PLUGINS, get_tools_result_async, function_call_list
from ..utils.scripts import check_json, safe_get, async_generator_to_sync
from ..core.request import prepare_request_payload

def get_filtered_keys_from_object(obj: object, *keys: str) -> Set[str]:
    """
    Get filtered list of object variable names.
    :param keys: List of keys to include. If the first key is "not", the remaining keys will be removed from the class keys.
    :return: List of class keys.
    """
    class_keys = obj.__dict__.keys()
    if not keys:
        return set(class_keys)

    # Remove the passed keys from the class keys.
    if keys[0] == "not":
        return {key for key in class_keys if key not in keys[1:]}
    # Check if all passed keys are valid
    if invalid_keys := set(keys) - class_keys:
        raise ValueError(
            f"Invalid keys: {invalid_keys}",
        )
    # Only return specified keys that are in class_keys
    return {key for key in keys if key in class_keys}

class chatgpt(BaseLLM):
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str = None,
        engine: str = os.environ.get("GPT_ENGINE") or "gpt-4o",
        api_url: str = (os.environ.get("API_URL") or "https://api.openai.com/v1/chat/completions"),
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
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
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        super().__init__(api_key, engine, api_url, system_prompt, proxy, timeout, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, reply_count, truncate_limit, use_plugins=use_plugins, print_log=print_log)
        self.conversation: dict[str, list[dict]] = {
            "default": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
            ],
        }
        self.function_calls_counter = {}
        self.function_call_max_loop = 3

        if self.tokens_usage["default"] > self.max_tokens:
            raise Exception("System prompt is too long")

    def add_to_conversation(
        self,
        message: Union[str, list],
        role: str,
        convo_id: str = "default",
        function_name: str = "",
        total_tokens: int = 0,
        function_arguments: str = "",
        pass_history: int = 9999,
        function_call_id: str = "",
    ) -> None:
        """
        Add a message to the conversation
        """
        # print("role", role, "function_name", function_name, "message", message)
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id)
        if function_name == "" and message and message != None:
            self.conversation[convo_id].append({"role": role, "content": message})
        elif function_name != "" and message and message != None:
            self.conversation[convo_id].append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": function_call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": function_arguments,
                        },
                    }
                ],
                })
            self.conversation[convo_id].append({"role": role, "tool_call_id": function_call_id, "content": message})
        else:
            print('\033[31m')
            print("error: add_to_conversation message is None or empty")
            print("role", role, "function_name", function_name, "message", message)
            print('\033[0m')

        conversation_len = len(self.conversation[convo_id]) - 1
        message_index = 0
        # print(json.dumps(self.conversation[convo_id], indent=4, ensure_ascii=False))
        while message_index < conversation_len:
            if self.conversation[convo_id][message_index]["role"] == self.conversation[convo_id][message_index + 1]["role"]:
                if self.conversation[convo_id][message_index].get("content") and self.conversation[convo_id][message_index + 1].get("content"):
                    if type(self.conversation[convo_id][message_index + 1]["content"]) == str \
                    and type(self.conversation[convo_id][message_index]["content"]) == list:
                        self.conversation[convo_id][message_index + 1]["content"] = [{"type": "text", "text": self.conversation[convo_id][message_index + 1]["content"]}]
                    if type(self.conversation[convo_id][message_index]["content"]) == str \
                    and type(self.conversation[convo_id][message_index + 1]["content"]) == list:
                        self.conversation[convo_id][message_index]["content"] = [{"type": "text", "text": self.conversation[convo_id][message_index]["content"]}]
                    self.conversation[convo_id][message_index]["content"] += self.conversation[convo_id][message_index + 1]["content"]
                self.conversation[convo_id].pop(message_index + 1)
                conversation_len = conversation_len - 1
            else:
                message_index = message_index + 1

        history_len = len(self.conversation[convo_id])

        history = pass_history
        if pass_history < 2:
            history = 2
        while history_len > history:
            mess_body = self.conversation[convo_id].pop(1)
            history_len = history_len - 1
            if mess_body.get("role") == "user":
                assistant_body = self.conversation[convo_id].pop(1)
                history_len = history_len - 1
                if assistant_body.get("tool_calls"):
                    self.conversation[convo_id].pop(1)
                    history_len = history_len - 1

        if total_tokens:
            self.tokens_usage[convo_id] += total_tokens

    def truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        while True:
            if (
                self.tokens_usage[convo_id] > self.truncate_limit
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                mess = self.conversation[convo_id].pop(1)
                print("Truncate message:", mess)
            else:
                break

    def extract_values(self, obj):
        if isinstance(obj, dict):
            for value in obj.values():
                yield from self.extract_values(value)
        elif isinstance(obj, list):
            for item in obj:
                yield from self.extract_values(item)
        else:
            yield obj

    async def get_post_body(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        **kwargs,
    ):
        self.conversation[convo_id][0] = {"role": "system","content": self.system_prompt}

        # 构造 provider 信息
        provider = {
            "provider": "openai",
            "base_url": kwargs.get('api_url', self.api_url.chat_url),
            "api": kwargs.get('api_key', self.api_key),
            "model": [model or self.engine],
            "tools": True if self.use_plugins else False,
            "image": True
        }

        # 构造请求数据
        request_data = {
            "model": model or self.engine,
            "messages": copy.deepcopy(self.conversation[convo_id]) if pass_history else [
                {"role": "system","content": self.system_prompt},
                {"role": role, "content": prompt}
            ],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
            "temperature": kwargs.get("temperature", self.temperature)
        }

        # 添加工具相关信息
        plugins = kwargs.get("plugins", PLUGINS)
        if not (all(value == False for value in plugins.values()) or self.use_plugins == False):
            tools = []
            # tools.append(copy.deepcopy(function_call_list["base"]))
            for item in plugins.keys():
                try:
                    if plugins[item]:
                        tools.append({"type": "function", "function": function_call_list[item]})
                except:
                    pass
            if tools:
                request_data["tools"] = tools
                request_data["tool_choice"] = "auto"

        # print("request_data", json.dumps(request_data, indent=4, ensure_ascii=False))

        # 调用核心模块的 prepare_request_payload 函数
        url, headers, json_post_body = await prepare_request_payload(provider, request_data)

        return json_post_body

    async def _process_stream_response(
        self,
        response_gen,
        convo_id="default",
        function_name="",
        total_tokens=0,
        function_arguments="",
        function_call_id="",
        model="",
        language="English",
        system_prompt=None,
        pass_history=9999,
        is_async=False,
        **kwargs
    ):
        """
        处理流式响应的共用逻辑

        :param response_gen: 响应生成器(同步或异步)
        :param is_async: 是否使用异步模式
        """
        response_role = None
        full_response = ""
        function_full_response = ""
        function_call_name = ""
        need_function_call = False

        # 处理单行数据的公共逻辑
        def process_line(line):
            nonlocal response_role, full_response, function_full_response, function_call_name, need_function_call, total_tokens, function_call_id

            if not line or (isinstance(line, str) and line.startswith(':')):
                return None

            if isinstance(line, str) and line.startswith('data:'):
                line = line.lstrip("data: ")
                if line == "[DONE]":
                    return "DONE"
            elif isinstance(line, (dict, list)):
                if isinstance(line, dict) and safe_get(line, "choices", 0, "message", "content"):
                    full_response = line["choices"][0]["message"]["content"]
                    return full_response
                else:
                    return str(line)
            else:
                try:
                    if isinstance(line, str):
                        line = json.loads(line)
                        if safe_get(line, "choices", 0, "message", "content"):
                            full_response = line["choices"][0]["message"]["content"]
                            return full_response
                        else:
                            return str(line)
                except:
                    print("json.loads error:", repr(line))
                    return None

            resp = json.loads(line) if isinstance(line, str) else line
            if "error" in resp:
                raise Exception(f"{resp}")

            total_tokens = total_tokens or safe_get(resp, "usage", "total_tokens", default=0)
            delta = safe_get(resp, "choices", 0, "delta")
            if not delta:
                return None

            response_role = response_role or safe_get(delta, "role")
            if safe_get(delta, "content"):
                need_function_call = False
                content = delta["content"]
                full_response += content
                return content

            if safe_get(delta, "tool_calls"):
                need_function_call = True
                function_call_name = function_call_name or safe_get(delta, "tool_calls", 0, "function", "name")
                function_full_response += safe_get(delta, "tool_calls", 0, "function", "arguments", default="")
                function_call_id = function_call_id or safe_get(delta, "tool_calls", 0, "id")
                return None

        # 处理流式响应
        async def process_async():
            nonlocal response_role, full_response, function_full_response, function_call_name, need_function_call, total_tokens, function_call_id

            async for line in response_gen:
                line = line.strip() if isinstance(line, str) else line
                result = process_line(line)
                if result == "DONE":
                    break
                elif result:
                    yield result

        def process_sync():
            nonlocal response_role, full_response, function_full_response, function_call_name, need_function_call, total_tokens, function_call_id

            for line in response_gen:
                line = line.decode("utf-8") if hasattr(line, "decode") else line
                result = process_line(line)
                if result == "DONE":
                    break
                elif result:
                    yield result

        # 使用同步或异步处理器处理响应
        if is_async:
            async for chunk in process_async():
                yield chunk
        else:
            for chunk in process_sync():
                yield chunk

        if self.print_log:
            print("\n\rtotal_tokens", total_tokens)

        if response_role is None:
            response_role = "assistant"

        # 处理函数调用
        if need_function_call:
            print("function_full_response", function_full_response)
            function_full_response = check_json(function_full_response)
            function_response = ""

            if not self.function_calls_counter.get(function_call_name):
                self.function_calls_counter[function_call_name] = 1
            else:
                self.function_calls_counter[function_call_name] += 1

            if self.function_calls_counter[function_call_name] <= self.function_call_max_loop and (function_full_response != "{}" or function_call_name == "get_date_time_weekday"):
                function_call_max_tokens = self.truncate_limit - 1000
                if function_call_max_tokens <= 0:
                    function_call_max_tokens = int(self.truncate_limit / 2)
                print("\033[32m function_call", function_call_name, "max token:", function_call_max_tokens, "\033[0m")

                # 处理函数调用结果
                if is_async:
                    async for chunk in get_tools_result_async(
                        function_call_name, function_full_response, function_call_max_tokens,
                        model or self.engine, chatgpt, kwargs.get('api_key', self.api_key),
                        self.api_url, use_plugins=False, model=model or self.engine,
                        add_message=self.add_to_conversation, convo_id=convo_id, language=language
                    ):
                        if "function_response:" in chunk:
                            function_response = chunk.replace("function_response:", "")
                        else:
                            yield chunk
                else:
                    async def run_async():
                        nonlocal function_response
                        async for chunk in get_tools_result_async(
                            function_call_name, function_full_response, function_call_max_tokens,
                            model or self.engine, chatgpt, kwargs.get('api_key', self.api_key),
                            self.api_url, use_plugins=False, model=model or self.engine,
                            add_message=self.add_to_conversation, convo_id=convo_id, language=language
                        ):
                            if "function_response:" in chunk:
                                function_response = chunk.replace("function_response:", "")
                            else:
                                yield chunk

                    for chunk in async_generator_to_sync(run_async()):
                        yield chunk
            else:
                function_response = "无法找到相关信息，停止使用 tools"

            response_role = "tool"

            # 递归处理函数调用响应
            if is_async:
                async for chunk in self.ask_stream_async(
                    function_response, response_role, convo_id=convo_id,
                    function_name=function_call_name, total_tokens=total_tokens,
                    model=model or self.engine, function_arguments=function_full_response,
                    function_call_id=function_call_id, api_key=kwargs.get('api_key', self.api_key),
                    plugins=kwargs.get("plugins", PLUGINS), system_prompt=system_prompt
                ):
                    yield chunk
            else:
                for chunk in self.ask_stream(
                    function_response, response_role, convo_id=convo_id,
                    function_name=function_call_name, total_tokens=total_tokens,
                    model=model or self.engine, function_arguments=function_full_response,
                    function_call_id=function_call_id, api_key=kwargs.get('api_key', self.api_key),
                    plugins=kwargs.get("plugins", PLUGINS), system_prompt=system_prompt
                ):
                    yield chunk
        else:
            # 添加响应到对话历史
            self.add_to_conversation(full_response, response_role, convo_id=convo_id, total_tokens=total_tokens, pass_history=pass_history)
            self.function_calls_counter = {}

            # 清理翻译引擎相关的历史记录
            if pass_history <= 2 and len(self.conversation[convo_id]) >= 2 \
            and (
                "You are a translation engine" in self.conversation[convo_id][-2]["content"] \
                or "You are a translation engine" in safe_get(self.conversation, convo_id, -2, "content", 0, "text", default="") \
                or "你是一位精通简体中文的专业翻译" in self.conversation[convo_id][-2]["content"] \
                or "你是一位精通简体中文的专业翻译" in safe_get(self.conversation, convo_id, -2, "content", 0, "text", default="")
            ):
                self.conversation[convo_id].pop(-1)
                self.conversation[convo_id].pop(-1)

    def ask_stream(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        function_name: str = "",
        total_tokens: int = 0,
        function_arguments: str = "",
        function_call_id: str = "",
        language: str = "English",
        system_prompt: str = None,
        **kwargs,
    ):
        """
        Ask a question (同步流式响应)
        """
        # 准备会话
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id, system_prompt=system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, function_name=function_name, total_tokens=total_tokens, function_arguments=function_arguments, function_call_id=function_call_id, pass_history=pass_history)

        # 获取请求体
        json_post = None
        async def get_post_body_async():
            nonlocal json_post
            json_post = await self.get_post_body(prompt, role, convo_id, model, pass_history, **kwargs)
            return json_post

        # 替换原来的获取请求体的代码
        # json_post = next(async_generator_to_sync(get_post_body_async()))
        try:
            json_post = asyncio.run(get_post_body_async())
        except RuntimeError:
            # 如果已经在事件循环中，则使用不同的方法
            loop = asyncio.get_event_loop()
            json_post = loop.run_until_complete(get_post_body_async())

        self.truncate_conversation(convo_id=convo_id)

        # 打印日志
        if self.print_log:
            print("api_url", kwargs.get('api_url', self.api_url.chat_url))
            print("api_key", kwargs.get('api_key', self.api_key))

        # 发送请求并处理响应
        for _ in range(3):
            if self.print_log:
                replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
                print(json.dumps(replaced_text, indent=4, ensure_ascii=False))

            response = None
            try:
                response = self.session.post(
                    kwargs.get('api_url', self.api_url.chat_url),
                    headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
                    json=json_post,
                    timeout=kwargs.get("timeout", self.timeout),
                    stream=True,
                )
            except ConnectionError:
                print("连接错误，请检查服务器状态或网络连接。")
                return
            except requests.exceptions.ReadTimeout:
                print("请求超时，请检查网络连接或增加超时时间。")
                return
            except Exception as e:
                print(f"发生了未预料的错误：{e}")
                if "Invalid URL" in str(e):
                    e = "You have entered an invalid API URL, please use the correct URL and use the `/start` command to set the API URL again. Specific error is as follows:\n\n" + str(e)
                    raise Exception(f"{e}")

            # 处理错误响应
            if response is not None:
                if response.status_code in (400, 422, 503):
                    json_post, should_retry = self._handle_response_error_sync(response, json_post)
                    if should_retry:
                        continue

                if response.status_code == 200:
                    if "is not possible because the prompts occupy" in response.text or response.text == "":
                        json_post, should_retry = self._handle_response_error_sync(response, json_post)
                        if should_retry:
                            continue
                    else:
                        break

        # 检查响应状态
        if response != None and response.status_code != 200:
            raise Exception(f"{response.status_code} {response.reason} {response.text[:400]}")
        if response is None:
            raise Exception(f"response is None, please check the connection or network.")

        # 处理响应流
        return async_generator_to_sync(self._process_stream_response(
            response.iter_lines(),
            convo_id=convo_id,
            function_name=function_name,
            total_tokens=total_tokens,
            function_arguments=function_arguments,
            function_call_id=function_call_id,
            model=model,
            language=language,
            system_prompt=system_prompt,
            pass_history=pass_history,
            is_async=False,
            **kwargs
        ))

    async def ask_stream_async(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        function_name: str = "",
        total_tokens: int = 0,
        function_arguments: str = "",
        function_call_id: str = "",
        language: str = "English",
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        **kwargs,
    ):
        """
        Ask a question (异步流式响应)
        """
        # 准备会话
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id, system_prompt=system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, function_name=function_name, total_tokens=total_tokens, function_arguments=function_arguments, pass_history=pass_history, function_call_id=function_call_id)

        # 获取请求体
        json_post = await self.get_post_body(prompt, role, convo_id, model, pass_history, **kwargs)
        self.truncate_conversation(convo_id=convo_id)

        # 打印日志
        if self.print_log:
            print("api_url", kwargs.get('api_url', self.api_url.chat_url))
            print("api_key", kwargs.get('api_key', self.api_key))

        # 发送请求并处理响应
        for _ in range(3):
            if self.print_log:
                replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
                print(json.dumps(replaced_text, indent=4, ensure_ascii=False))

            try:
                async with self.aclient.stream(
                    "post",
                    self.api_url.chat_url,
                    headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
                    json=json_post,
                    timeout=kwargs.get("timeout", self.timeout),
                ) as response:
                    if response is None:
                        raise Exception("Response is None, please check the connection or network.")

                    # 处理错误响应
                    if response.status_code in (400, 422, 503):
                        json_post, should_retry = await self._handle_response_error(response, json_post)
                        if should_retry:
                            continue

                    if response.status_code != 200:
                        await response.aread()
                        raise Exception(f"{response.status_code} {response.reason_phrase} {response.text[:400]}")

                    # 处理响应流
                    async for chunk in self._process_stream_response(
                        response.aiter_lines(),
                        convo_id=convo_id,
                        function_name=function_name,
                        total_tokens=total_tokens,
                        function_arguments=function_arguments,
                        function_call_id=function_call_id,
                        model=model,
                        language=language,
                        system_prompt=system_prompt,
                        pass_history=pass_history,
                        is_async=True,
                        **kwargs
                    ):
                        yield chunk

                break
            except Exception as e:
                print(f"发生了未预料的错误：{e}")
                import traceback
                traceback.print_exc()
                if "Invalid URL" in str(e):
                    e = "您输入了无效的API URL，请使用正确的URL并使用`/start`命令重新设置API URL。具体错误如下：\n\n" + str(e)
                    raise Exception(f"{e}")
                raise Exception(f"{e}")

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
        response = self.ask_stream_async(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            pass_history=pass_history,
            model=model or self.engine,
            **kwargs,
        )
        full_response: str = "".join([r async for r in response])
        return full_response

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally") -> None:
        """
        Reset the conversation
        """
        self.system_prompt = system_prompt or self.system_prompt
        self.conversation[convo_id] = [
            {"role": "system", "content": self.system_prompt},
        ]
        self.tokens_usage[convo_id] = 0

    def save(self, file: str, *keys: str) -> None:
        """
        Save the Chatbot configuration to a JSON file
        """
        with open(file, "w", encoding="utf-8") as f:
            data = {
                key: self.__dict__[key]
                for key in get_filtered_keys_from_object(self, *keys)
            }
            # saves session.proxies dict as session
            # leave this here for compatibility
            data["session"] = data["proxy"]
            del data["aclient"]
            json.dump(
                data,
                f,
                indent=2,
            )

    def load(self, file: Path, *keys_: str) -> None:
        """
        Load the Chatbot configuration from a JSON file
        """
        with open(file, encoding="utf-8") as f:
            # load json, if session is in keys, load proxies
            loaded_config = json.load(f)
            keys = get_filtered_keys_from_object(self, *keys_)

            if (
                "session" in keys
                and loaded_config["session"]
                or "proxy" in keys
                and loaded_config["proxy"]
            ):
                self.proxy = loaded_config.get("session", loaded_config["proxy"])
                self.session = httpx.Client(
                    follow_redirects=True,
                    proxies=self.proxy,
                    timeout=self.timeout,
                    cookies=self.session.cookies,
                    headers=self.session.headers,
                )
                self.aclient = httpx.AsyncClient(
                    follow_redirects=True,
                    proxies=self.proxy,
                    timeout=self.timeout,
                    cookies=self.session.cookies,
                    headers=self.session.headers,
                )
            if "session" in keys:
                keys.remove("session")
            if "aclient" in keys:
                keys.remove("aclient")
            self.__dict__.update({key: loaded_config[key] for key in keys})

    def _handle_response_error_common(self, response_text, json_post):
        """通用的响应错误处理逻辑，适用于同步和异步场景"""
        try:
            # 检查内容审核失败
            if "Content did not pass the moral check" in response_text:
                return json_post, False, f"内容未通过道德检查：{response_text[:400]}"

            # 处理函数调用相关错误
            if "function calling" in response_text:
                if "tools" in json_post:
                    del json_post["tools"]
                if "tool_choice" in json_post:
                    del json_post["tool_choice"]
                return json_post, True, None

            # 处理请求格式错误
            elif "invalid_request_error" in response_text:
                for index, mess in enumerate(json_post["messages"]):
                    if type(mess["content"]) == list and "text" in mess["content"][0]:
                        json_post["messages"][index] = {
                            "role": mess["role"],
                            "content": mess["content"][0]["text"]
                        }
                return json_post, True, None

            # 处理角色不允许错误
            elif "'function' is not an allowed role" in response_text:
                if json_post["messages"][-1]["role"] == "tool":
                    mess = json_post["messages"][-1]
                    json_post["messages"][-1] = {
                        "role": "assistant",
                        "name": mess["name"],
                        "content": mess["content"]
                    }
                return json_post, True, None

            # 处理服务器繁忙错误
            elif "Sorry, server is busy" in response_text:
                for index, mess in enumerate(json_post["messages"]):
                    if type(mess["content"]) == list and "text" in mess["content"][0]:
                        json_post["messages"][index] = {
                            "role": mess["role"],
                            "content": mess["content"][0]["text"]
                        }
                return json_post, True, None

            # 处理token超限错误
            elif "is not possible because the prompts occupy" in response_text:
                max_tokens = re.findall(r"only\s(\d+)\stokens", response_text)
                if max_tokens:
                    json_post["max_tokens"] = int(max_tokens[0])
                    return json_post, True, None

            # 默认移除工具相关设置
            else:
                if "tools" in json_post:
                    del json_post["tools"]
                if "tool_choice" in json_post:
                    del json_post["tool_choice"]
                return json_post, True, None

        except Exception as e:
            print(f"处理响应错误时出现异常: {e}")
            return json_post, False, str(e)

    def _handle_response_error_sync(self, response, json_post):
        """处理API响应错误并相应地修改请求体（同步版本）"""
        response_text = response.text

        # 处理空响应
        if response.status_code == 200 and response_text == "":
            for index, mess in enumerate(json_post["messages"]):
                if type(mess["content"]) == list and "text" in mess["content"][0]:
                    json_post["messages"][index] = {
                        "role": mess["role"],
                        "content": mess["content"][0]["text"]
                    }
            return json_post, True

        json_post, should_retry, error_msg = self._handle_response_error_common(response_text, json_post)

        if error_msg:
            raise Exception(f"{response.status_code} {response.reason} {error_msg}")

        return json_post, should_retry

    async def _handle_response_error(self, response, json_post):
        """处理API响应错误并相应地修改请求体（异步版本）"""
        await response.aread()
        response_text = response.text

        json_post, should_retry, error_msg = self._handle_response_error_common(response_text, json_post)

        if error_msg:
            raise Exception(f"{response.status_code} {response.reason_phrase} {error_msg}")

        return json_post, should_retry