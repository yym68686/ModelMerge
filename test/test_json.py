import json

# json_data = '爱'
# # json_data = '爱的主人，我会尽快为您规划一个走线到美国的安全路线。请您稍等片刻。\n\n首先，我会检查免签国家并为您提供相应的信息。接下来，我会 搜索有关旅行到美国的安全建议和路线规划。{}'

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

# 测试函数
input_string = '{"url": "https://github.com/fastai/fasthtml"'
result = split_json_strings(input_string)

for i, json_str in enumerate(result, 1):
    print(f"JSON {i}:", json_str)
    print("Parsed:", json.loads(json_str))
    print()

# def check_json(json_data):
#     while True:
#         try:
#             json.loads(json_data)
#             break
#         except json.decoder.JSONDecodeError as e:
#             print("JSON error：", e)
#             print("JSON body", repr(json_data))
#             if "Invalid control character" in str(e):
#                 json_data = json_data.replace("\n", "\\n")
#             if "Unterminated string starting" in str(e):
#                 json_data += '"}'
#             if "Expecting ',' delimiter" in str(e):
#                 json_data += '}'
#             if "Expecting value: line 1 column 1" in str(e):
#                 json_data = '{"prompt": ' + json.dumps(json_data) + '}'
#     return json_data
# print(json.loads(check_json(json_data)))

# a = '''
# '''

# print(json.loads(a))
