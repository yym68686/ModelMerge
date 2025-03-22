import json

class JSONExtractor:
    def __init__(self):
        self.buffer = ""
        self.bracket_count = 0
        self.in_target = False
        self.target_json = ""

    def process_line(self, line):
        self.buffer += line.strip()

        for char in line:
            if char == '{':
                self.bracket_count += 1
                if self.bracket_count == 4 and '"functionCall"' in self.buffer[-20:]:
                    self.in_target = True
                    self.target_json = '{'
            elif char == '}':
                if self.in_target:
                    self.target_json += '}'
                self.bracket_count -= 1
                if self.bracket_count == 3 and self.in_target:
                    self.in_target = False
                    return self.parse_target_json()

            if self.in_target:
                self.target_json += char

        return None

    def parse_target_json(self):
        try:
            parsed = json.loads(self.target_json)
            if 'functionCall' in parsed:
                return parsed['functionCall']
        except json.JSONDecodeError:
            pass
        return None

# 使用示例
extractor = JSONExtractor()

# 模拟流式接收数据
sample_lines = [
    '{\n',
    '  "candidates": [\n',
    '    {\n',
    '      "content": {\n',
    '        "parts": [\n',
    '          {\n',
    '            "functionCall": {\n',
    '              "name": "get_search_results",\n',
    '              "args": {\n',
    '                "prompt": "Claude Opus 3.5 release date"\n',
    '              }\n',
    '            }\n',
    '          }\n',
    '        ],\n',
    '        "role": "model"\n',
    '      },\n',
    '      "finishReason": "STOP",\n',
    '      "index": 0,\n',
    '      "safetyRatings": [\n',
    '        {\n',
    '          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",\n',
    '          "probability": "NEGLIGIBLE"\n',
    '        },\n',
    '        {\n',
    '          "category": "HARM_CATEGORY_HATE_SPEECH",\n',
    '          "probability": "NEGLIGIBLE"\n',
    '        },\n',
    '        {\n',
    '          "category": "HARM_CATEGORY_HARASSMENT",\n',
    '          "probability": "NEGLIGIBLE"\n',
    '        },\n',
    '        {\n',
    '          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",\n',
    '          "probability": "NEGLIGIBLE"\n',
    '        }\n',
    '      ]\n',
    '    }\n',
    '  ],\n',
    '  "usageMetadata": {\n',
    '    "promptTokenCount": 113,\n',
    '    "candidatesTokenCount": 55,\n',
    '    "totalTokenCount": 168\n',
    '  }\n',
    '}\n'
]

for line in sample_lines:
    result = extractor.process_line(line)
    if result:
        print("提取的functionCall:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        break