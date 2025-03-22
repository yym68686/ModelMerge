import os
import re
import json
import copy
import tiktoken
import requests

from .base import BaseLLM
from ..plugins import PLUGINS, get_tools_result_async, claude_tools_list
from ..utils.scripts import check_json, safe_get, async_generator_to_sync

class claudeConversation(dict):
    def Conversation(self, index):
        conversation_list = super().__getitem__(index)
        return "\n\n" + "\n\n".join([f"{item['role']}:{item['content']}" for item in conversation_list]) + "\n\nAssistant:"

class claude(BaseLLM):
    def __init__(
        self,
        api_key: str,
        engine: str = os.environ.get("GPT_ENGINE") or "claude-2.1",
        api_url: str = "https://api.anthropic.com/v1/complete",
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        temperature: float = 0.5,
        top_p: float = 0.7,
        timeout: float = 20,
        use_plugins: bool = True,
        print_log: bool = False,
    ):
        super().__init__(api_key, engine, api_url, system_prompt, timeout=timeout, temperature=temperature, top_p=top_p, use_plugins=use_plugins, print_log=print_log)
        # self.api_url = api_url
        self.conversation = claudeConversation()

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
        pass_history: int = 9999,
        total_tokens: int = 0,
    ) -> None:
        """
        Add a message to the conversation
        """

        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id)
        self.conversation[convo_id].append({"role": role, "content": message})

        history_len = len(self.conversation[convo_id])
        history = pass_history
        if pass_history < 2:
            history = 2
        while history_len > history:
            self.conversation[convo_id].pop(1)
            history_len = history_len - 1

        if total_tokens:
            self.tokens_usage[convo_id] += total_tokens

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = claudeConversation()
        self.system_prompt = system_prompt or self.system_prompt

    def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        while True:
            if (
                self.get_token_count(convo_id) > self.truncate_limit
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                self.conversation[convo_id].pop(1)
            else:
                break

    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        tiktoken.model.MODEL_TO_ENCODING["claude-2.1"] = "cl100k_base"
        encoding = tiktoken.encoding_for_model(self.engine)

        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 5
            for key, value in message.items():
                if value:
                    num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += 5  # role is always required and always 1 token
        num_tokens += 5  # every reply is primed with <im_start>assistant
        return num_tokens

    def ask_stream(
        self,
        prompt: str,
        role: str = "Human",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        model_max_tokens: int = 4096,
        **kwargs,
    ):
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, role, convo_id=convo_id, pass_history=pass_history)
        # self.__truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])

        url = self.api_url
        headers = {
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": f"{kwargs.get('api_key', self.api_key)}",
        }

        json_post = {
            "model": model or self.engine,
            "prompt": self.conversation.Conversation(convo_id) if pass_history else f"\n\nHuman:{prompt}\n\nAssistant:",
            "stream": True,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens_to_sample": model_max_tokens,
        }

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=json_post,
                timeout=kwargs.get("timeout", self.timeout),
                stream=True,
            )
        except ConnectionError:
            print("连接错误，请检查服务器状态或网络连接。")
            return
        except requests.exceptions.ReadTimeout:
            print("请求超时，请检查网络连接或增加超时时间。{e}")
            return
        except Exception as e:
            print(f"发生了未预料的错误: {e}")
            return

        if response.status_code != 200:
            print(response.text)
            raise BaseException(f"{response.status_code} {response.reason} {response.text}")
        response_role: str = "Assistant"
        full_response: str = ""
        for line in response.iter_lines():
            if not line or line.decode("utf-8") == "event: completion" or line.decode("utf-8") == "event: ping" or line.decode("utf-8") == "data: {}":
                continue
            line = line.decode("utf-8")[6:]
            # print(line)
            resp: dict = json.loads(line)
            content = resp.get("completion")
            if content:
                full_response += content
                yield content
        self.add_to_conversation(full_response, response_role, convo_id=convo_id, pass_history=pass_history)

class claude3(BaseLLM):
    def __init__(
        self,
        api_key: str = None,
        engine: str = os.environ.get("GPT_ENGINE") or "claude-3-5-sonnet-20241022",
        api_url: str = (os.environ.get("CLAUDE_API_URL") or "https://api.anthropic.com/v1/messages"),
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        temperature: float = 0.5,
        timeout: float = 20,
        top_p: float = 0.7,
        use_plugins: bool = True,
        print_log: bool = False,
    ):
        super().__init__(api_key, engine, api_url, system_prompt, timeout=timeout, temperature=temperature, top_p=top_p, use_plugins=use_plugins, print_log=print_log)
        self.conversation: dict[str, list[dict]] = {
            "default": [],
        }

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
        pass_history: int = 9999,
        total_tokens: int = 0,
        tools_id= "",
        function_name: str = "",
        function_full_response: str = "",
    ) -> None:
        """
        Add a message to the conversation
        """

        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id)
        if role == "user" or (role == "assistant" and function_full_response == ""):
            if type(message) == list:
                self.conversation[convo_id].append({
                    "role": role,
                    "content": message
                })
            if type(message) == str:
                self.conversation[convo_id].append({
                    "role": role,
                    "content": [{
                        "type": "text",
                        "text": message
                    }]
                })
        elif role == "assistant" and function_full_response:
            print("function_full_response", function_full_response)
            function_dict = {
                "type": "tool_use",
                "id": f"{tools_id}",
                "name": f"{function_name}",
                "input": json.loads(function_full_response)
                # "input": json.dumps(function_full_response, ensure_ascii=False)
            }
            self.conversation[convo_id].append({"role": role, "content": [function_dict]})
            function_dict = {
                "type": "tool_result",
                "tool_use_id": f"{tools_id}",
                "content": f"{message}",
                # "is_error": true
            }
            self.conversation[convo_id].append({"role": "user", "content": [function_dict]})

        conversation_len = len(self.conversation[convo_id]) - 1
        message_index = 0
        while message_index < conversation_len:
            if self.conversation[convo_id][message_index]["role"] == self.conversation[convo_id][message_index + 1]["role"]:
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
                mess_body = self.conversation[convo_id].pop(1)
                history_len = history_len - 1
                if safe_get(mess_body, "content", 0, "type") == "tool_use":
                    self.conversation[convo_id].pop(1)
                    history_len = history_len - 1

        if total_tokens:
            self.tokens_usage[convo_id] += total_tokens

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = list()
        self.system_prompt = system_prompt or self.system_prompt

    def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        while True:
            if (
                self.get_token_count(convo_id) > self.truncate_limit
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                self.conversation[convo_id].pop(1)
            else:
                break

    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        tiktoken.model.MODEL_TO_ENCODING["claude-2.1"] = "cl100k_base"
        encoding = tiktoken.encoding_for_model(self.engine)

        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 5
            for key, value in message.items():
                if value:
                    num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += 5  # role is always required and always 1 token
        num_tokens += 5  # every reply is primed with <im_start>assistant
        return num_tokens

    def ask_stream(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        model_max_tokens: int = 4096,
        tools_id: str = "",
        total_tokens: int = 0,
        function_name: str = "",
        function_full_response: str = "",
        language: str = "English",
        system_prompt: str = None,
        **kwargs,
    ):
        self.add_to_conversation(prompt, role, convo_id=convo_id, tools_id=tools_id, total_tokens=total_tokens, function_name=function_name, function_full_response=function_full_response, pass_history=pass_history)
        # self.__truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])

        url = self.api_url.source_api_url
        now_model = model or self.engine
        headers = {
            "content-type": "application/json",
            "x-api-key": f"{kwargs.get('api_key', self.api_key)}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15" if "claude-3-5-sonnet" in now_model else "tools-2024-05-16",
        }

        json_post = {
            "model": now_model,
            "messages": self.conversation[convo_id] if pass_history else [{
                "role": "user",
                "content": prompt
            }],
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": 8192 if "claude-3-5-sonnet" in now_model else model_max_tokens,
            "stream": True,
        }
        json_post["system"] = system_prompt or self.system_prompt
        plugins = kwargs.get("plugins", PLUGINS)
        if all(value == False for value in plugins.values()) == False and self.use_plugins:
            json_post.update(copy.deepcopy(claude_tools_list["base"]))
            for item in plugins.keys():
                try:
                    if plugins[item]:
                        json_post["tools"].append(claude_tools_list[item])
                except:
                    pass

        if self.print_log:
            replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
            print(json.dumps(replaced_text, indent=4, ensure_ascii=False))

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=json_post,
                timeout=kwargs.get("timeout", self.timeout),
                stream=True,
            )
        except ConnectionError:
            print("连接错误，请检查服务器状态或网络连接。")
            return
        except requests.exceptions.ReadTimeout:
            print("请求超时，请检查网络连接或增加超时时间。{e}")
            return
        except Exception as e:
            print(f"发生了未预料的错误: {e}")
            return

        if response.status_code != 200:
            print(response.text)
            raise BaseException(f"{response.status_code} {response.reason} {response.text}")
        response_role: str = "assistant"
        full_response: str = ""
        need_function_call: bool = False
        function_call_name: str = ""
        function_full_response: str = ""
        total_tokens = 0
        tools_id = ""
        for line in response.iter_lines():
            if not line or line.decode("utf-8")[:6] == "event:" or line.decode("utf-8") == "data: {}":
                continue
            # print(line.decode("utf-8"))
            # if "tool_use" in line.decode("utf-8"):
            #     tool_input = json.loads(line.decode("utf-8")["content"][1]["input"])
            # else:
            #     line = line.decode("utf-8")[6:]
            line = line.decode("utf-8")
            line = line.lstrip("data: ")
            # print(line)
            resp: dict = json.loads(line)
            if resp.get("error"):
                print("error:", resp["error"])
                raise BaseException(f"{resp['error']}")

            message = resp.get("message")
            if message:
                usage = message.get("usage")
                input_tokens = usage.get("input_tokens", 0)
                # output_tokens = usage.get("output_tokens", 0)
                output_tokens = 0
                total_tokens = total_tokens + input_tokens + output_tokens

            usage = resp.get("usage")
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                total_tokens = total_tokens + input_tokens + output_tokens

                # print("\n\rtotal_tokens", total_tokens)

            tool_use = resp.get("content_block")
            if tool_use and "tool_use" == tool_use['type']:
                # print("tool_use", tool_use)
                tools_id = tool_use["id"]
                need_function_call = True
                if "name" in tool_use:
                    function_call_name = tool_use["name"]
            delta = resp.get("delta")
            # print("delta", delta)
            if not delta:
                continue
            if "text" in delta:
                content = delta["text"]
                full_response += content
                yield content
            if "partial_json" in delta:
                function_call_content = delta["partial_json"]
                function_full_response += function_call_content

        # print("function_full_response", function_full_response)
        # print("function_call_name", function_call_name)
        # print("need_function_call", need_function_call)
        if self.print_log:
            print("\n\rtotal_tokens", total_tokens)
        if need_function_call:
            function_full_response = check_json(function_full_response)
            print("function_full_response", function_full_response)
            function_response = ""
            function_call_max_tokens = int(self.truncate_limit / 2)

            # function_response = yield from get_tools_result(function_call_name, function_full_response, function_call_max_tokens, self.engine, claude3, kwargs.get('api_key', self.api_key), self.api_url, use_plugins=False, model=model or self.engine, add_message=self.add_to_conversation, convo_id=convo_id, language=language)

            async def run_async():
                nonlocal function_response
                async for chunk in get_tools_result_async(
                    function_call_name, function_full_response, function_call_max_tokens,
                    model or self.engine, claude3, kwargs.get('api_key', self.api_key),
                    self.api_url, use_plugins=False, model=model or self.engine,
                    add_message=self.add_to_conversation, convo_id=convo_id, language=language
                ):
                    if "function_response:" in chunk:
                        function_response = chunk.replace("function_response:", "")
                    else:
                        yield chunk

            # 使用封装后的函数
            for chunk in async_generator_to_sync(run_async()):
                yield chunk

            response_role = "assistant"
            if self.conversation[convo_id][-1]["role"] == "function" and self.conversation[convo_id][-1]["name"] == "get_search_results":
                mess = self.conversation[convo_id].pop(-1)
            yield from self.ask_stream(function_response, response_role, convo_id=convo_id, function_name=function_call_name, total_tokens=total_tokens, model=model or self.engine, tools_id=tools_id, function_full_response=function_full_response, api_key=kwargs.get('api_key', self.api_key), plugins=kwargs.get("plugins", PLUGINS), system_prompt=system_prompt)
        else:
            if self.conversation[convo_id][-1]["role"] == "function" and self.conversation[convo_id][-1]["name"] == "get_search_results":
                mess = self.conversation[convo_id].pop(-1)
            self.add_to_conversation(full_response, response_role, convo_id=convo_id, total_tokens=total_tokens, pass_history=pass_history)
            self.function_calls_counter = {}
            if pass_history <= 2 and len(self.conversation[convo_id]) >= 2 and ("You are a translation engine" in self.conversation[convo_id][-2]["content"] or (type(self.conversation[convo_id][-2]["content"]) == list and "You are a translation engine" in self.conversation[convo_id][-2]["content"][0]["text"])):
                self.conversation[convo_id].pop(-1)
                self.conversation[convo_id].pop(-1)

    async def ask_stream_async(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        model_max_tokens: int = 4096,
        tools_id: str = "",
        total_tokens: int = 0,
        function_name: str = "",
        function_full_response: str = "",
        language: str = "English",
        system_prompt: str = None,
        **kwargs,
    ):
        self.add_to_conversation(prompt, role, convo_id=convo_id, tools_id=tools_id, total_tokens=total_tokens, function_name=function_name, function_full_response=function_full_response, pass_history=pass_history)
        # self.__truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])

        url = self.api_url.source_api_url
        now_model = model or self.engine
        headers = {
            "content-type": "application/json",
            "x-api-key": f"{kwargs.get('api_key', self.api_key)}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15" if "claude-3-5-sonnet" in now_model else "tools-2024-05-16",
        }

        json_post = {
            "model": now_model,
            "messages": self.conversation[convo_id] if pass_history else [{
                "role": "user",
                "content": prompt
            }],
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": 8192 if "claude-3-5-sonnet" in now_model else model_max_tokens,
            "stream": True,
        }
        json_post["system"] = system_prompt or self.system_prompt
        plugins = kwargs.get("plugins", PLUGINS)
        if all(value == False for value in plugins.values()) == False and self.use_plugins:
            json_post.update(copy.deepcopy(claude_tools_list["base"]))
            for item in plugins.keys():
                try:
                    if plugins[item]:
                        json_post["tools"].append(claude_tools_list[item])
                except:
                    pass

        if self.print_log:
            replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
            print(json.dumps(replaced_text, indent=4, ensure_ascii=False))

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=json_post,
                timeout=kwargs.get("timeout", self.timeout),
                stream=True,
            )
        except ConnectionError:
            print("连接错误，请检查服务器状态或网络连接。")
            return
        except requests.exceptions.ReadTimeout:
            print("请求超时，请检查网络连接或增加超时时间。{e}")
            return
        except Exception as e:
            print(f"发生了未预料的错误: {e}")
            return

        if response.status_code != 200:
            print(response.text)
            raise BaseException(f"{response.status_code} {response.reason} {response.text}")
        response_role: str = "assistant"
        full_response: str = ""
        need_function_call: bool = False
        function_call_name: str = ""
        function_full_response: str = ""
        total_tokens = 0
        tools_id = ""
        for line in response.iter_lines():
            if not line or line.decode("utf-8")[:6] == "event:" or line.decode("utf-8") == "data: {}":
                continue
            # print(line.decode("utf-8"))
            # if "tool_use" in line.decode("utf-8"):
            #     tool_input = json.loads(line.decode("utf-8")["content"][1]["input"])
            # else:
            #     line = line.decode("utf-8")[6:]
            line = line.decode("utf-8")[5:]
            if line.startswith(" "):
                line = line[1:]
            # print(line)
            resp: dict = json.loads(line)
            if resp.get("error"):
                print("error:", resp["error"])
                raise BaseException(f"{resp['error']}")

            message = resp.get("message")
            if message:
                usage = message.get("usage")
                input_tokens = usage.get("input_tokens", 0)
                # output_tokens = usage.get("output_tokens", 0)
                output_tokens = 0
                total_tokens = total_tokens + input_tokens + output_tokens

            usage = resp.get("usage")
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                total_tokens = total_tokens + input_tokens + output_tokens
                if self.print_log:
                    print("\n\rtotal_tokens", total_tokens)

            tool_use = resp.get("content_block")
            if tool_use and "tool_use" == tool_use['type']:
                # print("tool_use", tool_use)
                tools_id = tool_use["id"]
                need_function_call = True
                if "name" in tool_use:
                    function_call_name = tool_use["name"]
            delta = resp.get("delta")
            # print("delta", delta)
            if not delta:
                continue
            if "text" in delta:
                content = delta["text"]
                full_response += content
                yield content
            if "partial_json" in delta:
                function_call_content = delta["partial_json"]
                function_full_response += function_call_content
        # print("function_full_response", function_full_response)
        # print("function_call_name", function_call_name)
        # print("need_function_call", need_function_call)
        if need_function_call:
            function_full_response = check_json(function_full_response)
            print("function_full_response", function_full_response)
            function_response = ""
            function_call_max_tokens = int(self.truncate_limit / 2)
            async for chunk in get_tools_result_async(function_call_name, function_full_response, function_call_max_tokens, self.engine, claude3, kwargs.get('api_key', self.api_key), self.api_url, use_plugins=False, model=model or self.engine, add_message=self.add_to_conversation, convo_id=convo_id, language=language):
                if "function_response:" in chunk:
                    function_response = chunk.replace("function_response:", "")
                else:
                    yield chunk
            response_role = "assistant"
            if self.conversation[convo_id][-1]["role"] == "function" and self.conversation[convo_id][-1]["name"] == "get_search_results":
                mess = self.conversation[convo_id].pop(-1)
            async for chunk in self.ask_stream_async(function_response, response_role, convo_id=convo_id, function_name=function_call_name, total_tokens=total_tokens, model=model or self.engine, tools_id=tools_id, function_full_response=function_full_response, api_key=kwargs.get('api_key', self.api_key), plugins=kwargs.get("plugins", PLUGINS), system_prompt=system_prompt):
                yield chunk
            # yield from self.ask_stream(function_response, response_role, convo_id=convo_id, function_name=function_call_name, total_tokens=total_tokens, tools_id=tools_id, function_full_response=function_full_response)
        else:
            if self.conversation[convo_id][-1]["role"] == "function" and self.conversation[convo_id][-1]["name"] == "get_search_results":
                mess = self.conversation[convo_id].pop(-1)
            self.add_to_conversation(full_response, response_role, convo_id=convo_id, total_tokens=total_tokens, pass_history=pass_history)
            self.function_calls_counter = {}
            if pass_history <= 2 and len(self.conversation[convo_id]) >= 2 and ("You are a translation engine" in self.conversation[convo_id][-2]["content"] or (type(self.conversation[convo_id][-2]["content"]) == list and "You are a translation engine" in self.conversation[convo_id][-2]["content"][0]["text"])):
                self.conversation[convo_id].pop(-1)
                self.conversation[convo_id].pop(-1)