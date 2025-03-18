import os
import json
import base64
import tiktoken
import requests
import urllib.parse

def get_encode_text(text, model_name):
    tiktoken.get_encoding("cl100k_base")
    model_name = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(model_name)
    encode_text = encoding.encode(text, disallowed_special=())
    return encoding, encode_text

def get_text_token_len(text, model_name):
    encoding, encode_text = get_encode_text(text, model_name)
    return len(encode_text)

def cut_message(message: str, max_tokens: int, model_name: str):
    if type(message) != str:
        message = str(message)
    encoding, encode_text = get_encode_text(message, model_name)
    if len(encode_text) > max_tokens:
        encode_text = encode_text[:max_tokens]
        message = encoding.decode(encode_text)
    encode_text = encoding.encode(message)
    return message, len(encode_text)

import imghdr
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        file_content = image_file.read()
        file_type = imghdr.what(None, file_content)
        base64_encoded = base64.b64encode(file_content).decode('utf-8')

        if file_type == 'png':
            return f"data:image/png;base64,{base64_encoded}"
        elif file_type in ['jpeg', 'jpg']:
            return f"data:image/jpeg;base64,{base64_encoded}"
        else:
            raise ValueError(f"不支持的图片格式: {file_type}")

def get_doc_from_url(url):
    filename = urllib.parse.unquote(url.split("/")[-1])
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return filename

def get_encode_image(image_url):
    filename = get_doc_from_url(image_url)
    image_path = os.getcwd() + "/" + filename
    base64_image = encode_image(image_path)
    os.remove(image_path)
    return base64_image

from io import BytesIO
def get_audio_message(file_bytes):
    try:
        # 创建一个字节流对象
        audio_stream = BytesIO(file_bytes)

        # 直接使用字节流对象进行转录
        import config
        transcript = config.whisperBot.generate(audio_stream)
        # print("transcript", transcript)

        return transcript

    except Exception as e:
        return f"处理音频文件时出错： {str(e)}"

def Document_extract(docurl, docpath=None, engine = None):
    filename = docpath
    text = None
    prompt = None
    if docpath and docurl and "paper.pdf" != docpath:
        filename = get_doc_from_url(docurl)
        docpath = os.getcwd() + "/" + filename
    if filename and filename[-3:] == "pdf":
        from pdfminer.high_level import extract_text
        text = extract_text(docpath)
    if filename and (filename[-3:] == "txt" or filename[-3:] == ".md" or filename[-3:] == ".py" or filename[-3:] == "yml"):
        with open(docpath, 'r') as f:
            text = f.read()
    if text:
        prompt = (
            "Here is the document, inside <document></document> XML tags:"
            "<document>"
            "{}"
            "</document>"
        ).format(text)
    if filename and filename[-3:] == "jpg" or filename[-3:] == "png" or filename[-4:] == "jpeg":
        prompt = get_image_message(docurl, [], engine)
    if filename and filename[-3:] == "wav" or filename[-3:] == "mp3":
        with open(docpath, "rb") as file:
            file_bytes = file.read()
        prompt = get_audio_message(file_bytes)
        prompt = (
            "Here is the text content after voice-to-text conversion, inside <voice-to-text></voice-to-text> XML tags:"
            "<voice-to-text>"
            "{}"
            "</voice-to-text>"
        ).format(prompt)
    if os.path.exists(docpath):
        os.remove(docpath)
    return prompt

def split_json_strings(input_string):
    # 初始化结果列表和当前 JSON 字符串
    json_strings = []
    current_json = ""
    brace_count = 0

    # 遍历输入字符串的每个字符
    for char in input_string:
        current_json += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

            # 如果花括号配对完成，我们找到了一个完整的 JSON 字符串
            if brace_count == 0:
                # 尝试解析当前 JSON 字符串
                try:
                    json.loads(current_json)
                    json_strings.append(current_json)
                    current_json = ""
                except json.JSONDecodeError:
                    # 如果解析失败，继续添加字符
                    pass
    if json_strings == []:
        json_strings.append(input_string)
    return json_strings

def check_json(json_data):
    while True:
        try:
            result = split_json_strings(json_data)
            if len(result) > 0:
                json_data = result[0]
            json.loads(json_data)
            break
        except json.decoder.JSONDecodeError as e:
            print("JSON error：", e)
            print("JSON body", repr(json_data))
            if "Invalid control character" in str(e):
                json_data = json_data.replace("\n", "\\n")
            elif "Unterminated string starting" in str(e):
                json_data += '"}'
            elif "Expecting ',' delimiter" in str(e):
                json_data =  {"prompt": json_data}
            elif "Expecting ':' delimiter" in str(e):
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
            elif "Expecting value: line 1 column 1" in str(e):
                if json_data.startswith("prompt: "):
                    json_data = json_data.replace("prompt: ", "")
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
            else:
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
    return json_data

def is_surrounded_by_chinese(text, index):
    left_char = text[index - 1]
    if 0 < index < len(text) - 1:
        right_char = text[index + 1]
        return '\u4e00' <= left_char <= '\u9fff' or '\u4e00' <= right_char <= '\u9fff'
    if index == len(text) - 1:
        return '\u4e00' <= left_char <= '\u9fff'
    return False

def replace_char(string, index, new_char):
    return string[:index] + new_char + string[index+1:]

def claude_replace(text):
    Punctuation_mapping = {",": "，", ":": "：", "!": "！", "?": "？", ";": "；"}
    key_list = list(Punctuation_mapping.keys())
    for i in range(len(text)):
        if is_surrounded_by_chinese(text, i) and (text[i] in key_list):
            text = replace_char(text, i, Punctuation_mapping[text[i]])
    return text

def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key] if isinstance(data, (dict, list)) else data.get(key)
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    return data

import asyncio
def async_generator_to_sync(async_gen):
    """
    将异步生成器转换为同步生成器的工具函数

    Args:
        async_gen: 异步生成器函数

    Yields:
        异步生成器产生的每个值
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        async def collect_chunks():
            chunks = []
            async for chunk in async_gen:
                chunks.append(chunk)
            return chunks

        chunks = loop.run_until_complete(collect_chunks())
        for chunk in chunks:
            yield chunk

    except Exception as e:
        print(f"Error during async execution: {e}")
        raise
    finally:
        try:
            # 清理所有待处理的任务
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    os.system("clear")