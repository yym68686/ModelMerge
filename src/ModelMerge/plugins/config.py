import os
import json
import inspect

# 明确导入需要的函数，而不是使用通配符导入
from .websearch import get_search_results, get_url_content
from .arXiv import download_read_arxiv_pdf
from .image import generate_image
from .today import get_date_time_weekday
from .run_python import run_python_script

from ..utils.scripts import cut_message
from ..utils.prompt import search_key_word_prompt, arxiv_doc_user_prompt

def function_to_json(func) -> dict:
    """
    将Python函数转换为JSON可序列化的字典，描述函数的签名，包括名称、描述和参数。

    Args:
        func: 要转换的函数

    Returns:
        表示函数签名的JSON格式字典
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"获取函数{func.__name__}签名失败: {str(e)}")

    parameters = {}
    for param in signature.parameters.values():
        try:
            if param.annotation == inspect._empty:
                parameters[param.name] = {"type": "string"}
            else:
                parameters[param.name] = {"type": type_map.get(param.annotation, "string")}
        except KeyError as e:
            raise KeyError(f"未知类型注解 {param.annotation} 用于参数 {param.name}: {str(e)}")

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

def gpt2claude_tools_json(json_dict):
    import copy
    json_dict = copy.deepcopy(json_dict)
    keys_to_change = {
        "parameters": "input_schema",
    }
    for old_key, new_key in keys_to_change.items():
        if old_key in json_dict:
            if new_key:
                json_dict[new_key] = json_dict.pop(old_key)
            else:
                json_dict.pop(old_key)
        else:
            if new_key and "description" in json_dict.keys():
                json_dict[new_key] = {
                    "type": "object",
                    "properties": {}
                }
    if "tools" in json_dict.keys():
        json_dict["tool_choice"] = {
            "type": "auto"
        }
    return json_dict

async def get_tools_result_async(function_call_name, function_full_response, function_call_max_tokens, engine, robot, api_key, api_url, use_plugins, model, add_message, convo_id, language):
    function_response = ""
    if function_call_name == "get_search_results":
        prompt = json.loads(function_full_response)["query"]
        yield "message_search_stage_1"
        llm = robot(api_key=api_key, api_url=api_url.source_api_url, engine=engine, use_plugins=use_plugins)
        keywords = (await llm.ask_async(search_key_word_prompt.format(source=prompt), model=model)).split("\n")
        print("keywords", keywords)
        keywords = [item.replace("三行关键词是：", "") for item in keywords if "\\x" not in item if item != ""]
        keywords = [prompt] + keywords
        keywords = keywords[:3]
        print("select keywords", keywords)
        async for chunk in get_search_results(keywords):
            if type(chunk) == str:
                yield chunk
            else:
                function_response = "\n\n".join(chunk)
            # function_response = yield chunk
        # function_response = yield from eval(function_call_name)(prompt, keywords)
        function_call_max_tokens = 32000
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
        if function_response:
            function_response = (
                f"You need to response the following question: {prompt}. Search results is provided inside <Search_results></Search_results> XML tags. Your task is to think about the question step by step and then answer the above question in {language} based on the Search results provided. Please response in {language} and adopt a style that is logical, in-depth, and detailed. Note: In order to make the answer appear highly professional, you should be an expert in textual analysis, aiming to make the answer precise and comprehensive. Directly response markdown format, without using markdown code blocks. For each sentence quoting search results, a markdown ordered superscript number url link must be used to indicate the source, e.g., [¹](https://www.example.com)"
                "Here is the Search results, inside <Search_results></Search_results> XML tags:"
                "<Search_results>"
                "{}"
                "</Search_results>"
            ).format(function_response)
        else:
            function_response = "无法找到相关信息，停止使用 tools"
        # user_prompt = f"You need to response the following question: {prompt}. Search results is provided inside <Search_results></Search_results> XML tags. Your task is to think about the question step by step and then answer the above question in {config.language} based on the Search results provided. Please response in {config.language} and adopt a style that is logical, in-depth, and detailed. Note: In order to make the answer appear highly professional, you should be an expert in textual analysis, aiming to make the answer precise and comprehensive. Directly response markdown format, without using markdown code blocks"
        # self.add_to_conversation(user_prompt, "user", convo_id=convo_id)
    if function_call_name == "get_url_content":
        url = json.loads(function_full_response)["url"]
        print("\n\nurl", url)
        function_response = get_url_content(url)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    if function_call_name == "generate_image":
        prompt = json.loads(function_full_response)["text"]
        function_response = generate_image(prompt)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    if function_call_name == "download_read_arxiv_pdf":
        add_message(arxiv_doc_user_prompt, "user", convo_id=convo_id)
        # add_message(arxiv_doc_assistant_prompt, "assistant", convo_id=convo_id)
        prompt = json.loads(function_full_response)["arxiv_id"]
        function_response = download_read_arxiv_pdf(prompt)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    if function_call_name == "run_python_script":
        prompt = json.loads(function_full_response)["code"]
        function_response = await run_python_script(prompt)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    if function_call_name == "get_date_time_weekday":
        function_response = get_date_time_weekday()
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)

    function_response = (
        f"function_response:{function_response}"
    )
    yield function_response
    # return function_response

PLUGINS = {
    "SEARCH" : (os.environ.get('SEARCH', "True") == "False") == False,
    "URL"    : (os.environ.get('URL', "True") == "False") == False,
    "ARXIV"  : (os.environ.get('ARXIV', "False") == "False") == False,
    "CODE"   : (os.environ.get('CODE', "False") == "False") == False,
    "IMAGE"  : (os.environ.get('IMAGE', "False") == "False") == False,
    "DATE"   : (os.environ.get('DATE', "False") == "False") == False,
}

function_call_list = {
    "SEARCH": function_to_json(get_search_results)["function"],
    "URL": function_to_json(get_url_content)["function"],
    "IMAGE": function_to_json(generate_image)["function"],
    "ARXIV": function_to_json(download_read_arxiv_pdf)["function"],
    "CODE": function_to_json(run_python_script)["function"],
    "DATE": function_to_json(get_date_time_weekday)["function"],
}

claude_tools_list = {f"{key}": gpt2claude_tools_json(function_call_list[key]) for key in function_call_list.keys()}