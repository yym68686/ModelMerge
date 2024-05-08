from tools import claude_tools_list

class claudeConversation(dict):
    def Conversation(self, index):
        conversation_list = super().__getitem__(index)
        return "\n\n" + "\n\n".join([f"{item['role']}:{item['content']}" for item in conversation_list]) + "\n\nAssistant:"

class claudebot:
    def __init__(
        self,
        api_key: str,
        engine: str = os.environ.get("GPT_ENGINE") or "claude-2.1",
        temperature: float = 0.5,
        top_p: float = 0.7,
        chat_url: str = "https://api.anthropic.com/v1/complete",
        timeout: float = 20,
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        **kwargs,
    ):
        self.api_key: str = api_key
        self.engine: str = engine
        self.temperature = temperature
        self.top_p = top_p
        self.chat_url = chat_url
        self.timeout = timeout
        self.session = requests.Session()
        self.conversation = claudeConversation()
        self.system_prompt = system_prompt

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
        pass_history: bool = True,
    ) -> None:
        """
        Add a message to the conversation
        """

        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id)
        self.conversation[convo_id].append({"role": role, "content": message})

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = list()

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
        if self.engine not in ENGINES:
            raise NotImplementedError(
                f"Engine {self.engine} is not supported. Select from {ENGINES}",
            )
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
        model: str = None,
        pass_history: bool = True,
        model_max_tokens: int = 4096,
        **kwargs,
    ):
        pass_history = True
        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, role, convo_id=convo_id)
        # self.__truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])

        url = self.chat_url
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
        self.add_to_conversation(full_response, response_role, convo_id=convo_id)
        # print(repr(self.conversation.Conversation(convo_id)))
        # print("total tokens:", self.get_token_count(convo_id))

class claude3bot:
    def __init__(
        self,
        api_key: str,
        engine: str = os.environ.get("GPT_ENGINE") or "claude-3-opus-20240229",
        temperature: float = 0.5,
        top_p: float = 0.7,
        chat_url: str = "https://api.anthropic.com/v1/messages",
        timeout: float = 20,
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        **kwargs,
    ):
        self.api_key: str = api_key
        self.engine: str = engine
        self.temperature = temperature
        self.top_p = top_p
        self.chat_url = chat_url
        self.timeout = timeout
        self.session = requests.Session()
        self.conversation: dict[str, list[dict]] = {
            "default": [],
        }
        self.system_prompt = system_prompt

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
        pass_history: bool = True,
    ) -> None:
        """
        Add a message to the conversation
        """

        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id)
        # print("message", message)
        self.conversation[convo_id].append({"role": role, "content": message})
        index = len(self.conversation[convo_id]) - 2
        if index >= 0 and self.conversation[convo_id][index]["role"] == self.conversation[convo_id][index + 1]["role"]:
            self.conversation[convo_id][index]["content"] += self.conversation[convo_id][index + 1]["content"]
            self.conversation[convo_id].pop(index + 1)

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = list()

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
        if self.engine not in ENGINES:
            raise NotImplementedError(
                f"Engine {self.engine} is not supported. Select from {ENGINES}",
            )
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
        model: str = None,
        pass_history: bool = True,
        model_max_tokens: int = 4096,
        **kwargs,
    ):
        pass_history = True
        if convo_id not in self.conversation or pass_history == False:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, role, convo_id=convo_id)
        # self.__truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])

        url = self.chat_url
        headers = {
            "x-api-key": f"{kwargs.get('api_key', self.api_key)}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            # "anthropic-beta": "tools-2024-04-04"
        }

        json_post = {
            "model": model or self.engine,
            "messages": self.conversation[convo_id] if pass_history else [{
                "role": "user",
                "content": prompt
            }],
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": model_max_tokens,
            "stream": True,
        }
        if self.system_prompt:
            json_post["system"] = self.system_prompt

        print(json.dumps(json_post, indent=4, ensure_ascii=False))

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
            if not line or line.decode("utf-8")[:6] == "event:" or line.decode("utf-8") == "data: {}":
                continue
            # print(line.decode("utf-8"))
            # if "tool_use" in line.decode("utf-8"):
            #     tool_input = json.loads(line.decode("utf-8")["content"][1]["input"])
            # else:
            #     line = line.decode("utf-8")[6:]
            line = line.decode("utf-8")[6:]
            # print(line)
            resp: dict = json.loads(line)
            delta = resp.get("delta")
            if not delta:
                continue
            if "text" in delta:
                content = delta["text"]
                full_response += content
                yield content
        self.add_to_conversation(full_response, response_role, convo_id=convo_id)
        # print(repr(self.conversation.Conversation(convo_id)))
        # print("total tokens:", self.get_token_count(convo_id))