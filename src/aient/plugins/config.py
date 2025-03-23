import os
import json
import inspect

from .registry import registry
from ..utils.scripts import cut_message
from ..utils.prompt import search_key_word_prompt, arxiv_doc_user_prompt

async def get_tools_result_async(function_call_name, function_full_response, function_call_max_tokens, engine, robot, api_key, api_url, use_plugins, model, add_message, convo_id, language):
    function_response = ""
    if function_call_name in registry.tools:
        function_to_call = registry.tools[function_call_name]
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
        async for chunk in function_to_call(keywords):
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
    else:
        prompt = json.loads(function_full_response)
        if inspect.iscoroutinefunction(function_to_call):
            function_response = await function_to_call(**prompt)
        else:
            function_response = function_to_call(**prompt)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)

    if function_call_name == "download_read_arxiv_pdf":
        add_message(arxiv_doc_user_prompt, "user", convo_id=convo_id)

    function_response = (
        f"function_response:{function_response}"
    )
    yield function_response
    # return function_response

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
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": required,
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

# print("registry.tools", json.dumps(registry.tools_info.get('get_time', {}), indent=4, ensure_ascii=False))
# print("registry.tools", json.dumps(registry.tools_info['run_python_script'].to_dict(), indent=4, ensure_ascii=False))

# 修改PLUGINS定义，使用registry中的工具
def get_plugins():
    return {
        tool_name: (os.environ.get(tool_name, "False") == "False") == False
        for tool_name in registry.tools.keys()
    }

# 修改function_call_list定义，使用registry中的工具
def get_function_call_list():
    function_list = {}
    for tool_name, tool_func in registry.tools.items():
        function_list[tool_name] = function_to_json(tool_func)
    return function_list

def get_claude_tools_list():
    function_list = get_function_call_list()
    return {f"{key}": gpt2claude_tools_json(function_list[key]) for key in function_list.keys()}

# 初始化默认配置
PLUGINS = get_plugins()
function_call_list = get_function_call_list()
claude_tools_list = get_claude_tools_list()

# 动态更新工具函数配置
def update_tools_config():
    global PLUGINS, function_call_list, claude_tools_list
    PLUGINS = get_plugins()
    function_call_list = get_function_call_list()
    claude_tools_list = get_claude_tools_list()
    return PLUGINS, function_call_list, claude_tools_list