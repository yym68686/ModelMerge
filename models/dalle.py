class Imagebot:
    def __init__(
        self,
        api_key: str,
        timeout: float = 20,
    ):
        self.api_key: str = api_key
        self.engine: str = "dall-e-3"
        self.session = requests.Session()
        self.timeout: float = timeout

    def dall_e_3(
        self,
        prompt: str,
        model: str = None,
        **kwargs,
    ):
        url = config.bot_api_url.image_url
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
            raise t.APIConnectionError(
                f"{response.status_code} {response.reason} {response.text}",
            )
        json_data = json.loads(response.text)
        url = json_data["data"][0]["url"]
        yield url