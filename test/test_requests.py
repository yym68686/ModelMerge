import json
import requests

class APIClient:
    def __init__(self, api_url, api_key, timeout=30):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def post_request(self, json_post, **kwargs):
        for _ in range(2):
            self.print_json(json_post)
            try:
                response = self.send_post_request(json_post, **kwargs)
            except (ConnectionError, requests.exceptions.ReadTimeout, Exception) as e:
                self.handle_exception(e)
                return
            if response.status_code == 400:
                self.handle_bad_request(response, json_post)
                continue
            if response.status_code == 200:
                break
        if response.status_code != 200:
            raise Exception(f"{response.status_code} {response.reason} {response.text}")

    def print_json(self, json_post):
        print(json.dumps(json_post, indent=4, ensure_ascii=False))

    def send_post_request(self, json_post, **kwargs):
        return self.session.post(
            self.api_url.chat_url,
            headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
            json=json_post,
            timeout=kwargs.get("timeout", self.timeout),
            stream=True,
        )

    def handle_exception(self, e):
        if isinstance(e, ConnectionError):
            print("连接错误，请检查服务器状态或网络连接。")
        elif isinstance(e, requests.exceptions.ReadTimeout):
            print("请求超时，请检查网络连接或增加超时时间。")
        else:
            print(f"发生了未预料的错误: {e}")

    def handle_bad_request(self, response, json_post):
        print("response.text", response.text)
        if "invalid_request_error" in response.text:
            self.fix_invalid_request_error(json_post)
        else:
            self.remove_unnecessary_fields(json_post)

    def fix_invalid_request_error(self, json_post):
        for index, mess in enumerate(json_post["messages"]):
            if isinstance(mess["content"], list):
                json_post["messages"][index] = {
                    "role": mess["role"],
                    "content": mess["content"][0]["text"]
                }

    def remove_unnecessary_fields(self, json_post):
        if "function_call" in json_post:
            del json_post["function_call"]
        if "functions" in json_post:
            del json_post["functions"]

# Usage
api_client = APIClient(api_url="https://api.example.com", api_key="your_api_key")
json_post = {
    "messages": [
        {"role": "user", "content": "Hello"}
    ]
}
api_client.post_request(json_post)



import json

def process_line(line):
    """处理每一行数据."""
    if not line or line.startswith(':'):
        return None
    if line.startswith('data:'):
        return line[6:]
    return line

def parse_response(line):
    """解析响应行."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None

def handle_usage(usage):
    """处理使用情况."""
    total_tokens = usage.get("total_tokens", 0)
    print("\n\rtotal_tokens", total_tokens)

def handle_choices(choices, full_response, response_role, need_function_call, function_full_response):
    """处理响应中的选择."""
    delta = choices[0].get("delta")
    if not delta:
        return full_response, response_role, need_function_call, function_full_response

    if "role" in delta and response_role is None:
        response_role = delta["role"]
    if "content" in delta and delta["content"]:
        need_function_call = False
        content = delta["content"]
        full_response += content
        yield content
    if "function_call" in delta:
        need_function_call = True
        function_call_content = delta["function_call"]["arguments"]
        if "name" in delta["function_call"]:
            function_call_name = delta["function_call"]["name"]
        function_full_response += function_call_content
        if function_full_response.count("\\n") > 2 or "}" in function_full_response:
            return full_response, response_role, need_function_call, function_full_response

    return full_response, response_role, need_function_call, function_full_response

def read_http_stream(response):
    """读取 HTTP 流并处理数据."""
    full_response = ""
    function_full_response = ""
    response_role = None
    need_function_call = False

    for line in response.iter_lines():
        line = line.decode("utf-8")
        processed_line = process_line(line)
        if processed_line is None:
            continue

        if processed_line == "[DONE]":
            break

        resp = parse_response(processed_line)
        if resp is None:
            continue

        usage = resp.get("usage")
        if usage:
            handle_usage(usage)

        choices = resp.get("choices")
        if not choices:
            continue

        result = handle_choices(choices, full_response, response_role, need_function_call, function_full_response)
        if isinstance(result, tuple):
            full_response, response_role, need_function_call, function_full_response = result
        else:
            yield result

# 使用示例
# response = requests.get("your_streaming_api_url", stream=True)
# for content in read_http_stream(response):
#     print(content)