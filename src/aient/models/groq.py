import os
import json
import requests
import tiktoken

from .base import BaseLLM

class groq(BaseLLM):
    def __init__(
        self,
        api_key: str = None,
        engine: str = os.environ.get("GPT_ENGINE") or "llama3-70b-8192",
        api_url: str = "https://api.groq.com/openai/v1/chat/completions",
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        temperature: float = 0.5,
        top_p: float = 1,
        timeout: float = 20,
    ):
        super().__init__(api_key, engine, api_url, system_prompt, timeout=timeout, temperature=temperature, top_p=top_p)
        self.api_url = api_url

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
        # tiktoken.model.MODEL_TO_ENCODING["mixtral-8x7b-32768"] = "cl100k_base"
        encoding = tiktoken.get_encoding("cl100k_base")

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
        model_max_tokens: int = 1024,
        system_prompt: str = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, role, convo_id=convo_id, pass_history=pass_history)
        # self.__truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])

        url = self.api_url
        headers = {
            "Authorization": f"Bearer {kwargs.get('GROQ_API_KEY', self.api_key)}",
            "Content-Type": "application/json",
        }

        self.conversation[convo_id][0] = {"role": "system","content": self.system_prompt}
        json_post = {
            "messages": self.conversation[convo_id] if pass_history else [{
                "role": "user",
                "content": prompt
            }],
            "model": model or self.engine,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": model_max_tokens,
            "top_p": kwargs.get("top_p", self.top_p),
            "stop": None,
            "stream": True,
        }
        # print("json_post", json_post)
        # print(os.environ.get("GPT_ENGINE"), model, self.engine)

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
        for line in response.iter_lines():
            if not line:
                continue
            # Remove "data: "
            # print(line.decode("utf-8"))
            if line.decode("utf-8")[:6] == "data: ":
                line = line.decode("utf-8")[6:]
            else:
                print(line.decode("utf-8"))
                full_response = json.loads(line.decode("utf-8"))["choices"][0]["message"]["content"]
                yield full_response
                break
            if line == "[DONE]":
                break
            resp: dict = json.loads(line)
            # print("resp", resp)
            choices = resp.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta")
            if not delta:
                continue
            if "role" in delta:
                response_role = delta["role"]
            if "content" in delta and delta["content"]:
                content = delta["content"]
                full_response += content
                yield content
        self.add_to_conversation(full_response, response_role, convo_id=convo_id, pass_history=pass_history)

    async def ask_stream_async(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        model_max_tokens: int = 1024,
        system_prompt: str = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, role, convo_id=convo_id, pass_history=pass_history)
        # self.__truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])

        url = self.api_url
        headers = {
            "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', self.api_key) or kwargs.get('api_key')}",
            "Content-Type": "application/json",
        }

        self.conversation[convo_id][0] = {"role": "system","content": self.system_prompt}
        json_post = {
            "messages": self.conversation[convo_id] if pass_history else [{
                "role": "user",
                "content": prompt
            }],
            "model": model or self.engine,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": model_max_tokens,
            "top_p": kwargs.get("top_p", self.top_p),
            "stop": None,
            "stream": True,
        }
        # print("json_post", json_post)
        # print(os.environ.get("GPT_ENGINE"), model, self.engine)

        response_role: str = "assistant"
        full_response: str = ""
        try:
            async with self.aclient.stream(
                "post",
                url,
                headers=headers,
                json=json_post,
                timeout=kwargs.get("timeout", self.timeout),
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    print(response.text)
                    raise BaseException(f"{response.status_code} {response.reason} {response.text}")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    # Remove "data: "
                    # print(line)
                    if line[:6] == "data: ":
                        line = line.lstrip("data: ")
                    else:
                        full_response = json.loads(line)["choices"][0]["message"]["content"]
                        yield full_response
                        break
                    if line == "[DONE]":
                        break
                    resp: dict = json.loads(line)
                    # print("resp", resp)
                    choices = resp.get("choices")
                    if not choices:
                        continue
                    delta = choices[0].get("delta")
                    if not delta:
                        continue
                    if "role" in delta:
                        response_role = delta["role"]
                    if "content" in delta and delta["content"]:
                        content = delta["content"]
                        full_response += content
                        yield content
        except Exception as e:
            print(f"发生了未预料的错误: {e}")
            import traceback
            traceback.print_exc()
            return

        self.add_to_conversation(full_response, response_role, convo_id=convo_id, pass_history=pass_history)