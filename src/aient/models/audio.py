import os
import requests
import json
from .base import BaseLLM

API = os.environ.get('API', None)
API_URL = os.environ.get('API_URL', None)

class whisper(BaseLLM):
    def __init__(
        self,
        api_key: str,
        api_url: str = (os.environ.get("API_URL") or "https://api.openai.com/v1/audio/transcriptions"),
        timeout: float = 20,
    ):
        super().__init__(api_key, api_url=api_url, timeout=timeout)
        self.engine: str = "whisper-1"

    def generate(
        self,
        audio_file: bytes,
        model: str = "whisper-1",
        **kwargs,
    ):
        url = self.api_url.audio_transcriptions
        headers = {"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"}

        files = {
            "file": ("audio.mp3", audio_file, "audio/mpeg")
        }

        data = {
            "model": os.environ.get("AUDIO_MODEL_NAME") or model or self.engine,
        }
        try:
            response = self.session.post(
                url,
                headers=headers,
                data=data,
                files=files,
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
        text = json_data["text"]
        return text

def audio_transcriptions(text):
    dallbot = whisper(api_key=f"{API}")
    for data in dallbot.generate(text):
        return data