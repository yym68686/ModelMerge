import os
import requests
import json
from ..models.base import BaseLLM
from .registry import register_tool

API = os.environ.get('API', None)
API_URL = os.environ.get('API_URL', None)

class dalle3(BaseLLM):
    def __init__(
        self,
        api_key: str,
        api_url: str = (os.environ.get("API_URL") or "https://api.openai.com/v1/images/generations"),
        timeout: float = 20,
    ):
        super().__init__(api_key, api_url=api_url, timeout=timeout)
        self.engine: str = "dall-e-3"

    def generate(
        self,
        prompt: str,
        model: str = "",
        **kwargs,
    ):
        url = self.api_url.image_url
        headers = {"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"}

        json_post = {
                "model": os.environ.get("IMAGE_MODEL_NAME") or model or self.engine,
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
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
            raise Exception(f"{response.status_code} {response.reason} {response.text}")
        json_data = json.loads(response.text)
        url = json_data["data"][0]["url"]
        yield url

@register_tool()
def generate_image(text):
    """
    生成图像

    参数:
        text: 描述图像的文本

    返回:
        图像的URL
    """
    dallbot = dalle3(api_key=f"{API}")
    for data in dallbot.generate(text):
        return data