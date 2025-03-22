# import os
# import sys
# print(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .chatgpt import function_call_list
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

claude_tools_list = {f"{key}": gpt2claude_tools_json(function_call_list[key]) for key in function_call_list.keys()}
if __name__ == "__main__":
    print(claude_tools_list)