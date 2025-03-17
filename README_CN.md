# modelmerge

[英文](./README.md) | [中文](./README_CN.md)

modelmerge 是一个强大的库，旨在简化和统一不同大型语言模型的使用，包括 GPT-3.5/4/4 Turbo/4o、o1-preview/o1-mini、DALL-E 3、Claude2/3/3.5、Gemini1.5 Pro/Flash、Vertex AI(Claude, Gemini) 、DuckDuckGo 和 Groq。该库支持 GPT 格式的函数调用，并内置了 Google 搜索和 URL 总结功能，极大地增强了模型的实用性和灵活性。

## ✨ 特性

- **多模型支持**：集成多种最新的大语言模型。
- **实时交互**：支持实时查询流，实时获取模型响应。
- **功能扩展**：通过内置的函数调用（function calling）支持，可以轻松扩展模型的功能，目前支持 DuckDuckGo 和 Google 搜索、内容摘要、Dalle-3画图、arXiv 论文总结、当前时间、代码解释器等插件。
- **简易接口**：提供简洁统一的 API 接口，使得调用和管理模型变得轻松。

## 快速上手

以下是如何在您的 Python 项目中快速集成和使用 modelmerge 的指南。

### 安装

首先，您需要安装 modelmerge。可以通过 pip 直接安装：

```bash
pip install modelmerge
```

### 使用示例

以下是一个简单的示例，展示如何使用 modelmerge 来请求 GPT-4 模型并处理返回的流式数据：

```python
from ModelMerge import chatgpt

# 初始化模型，设置 API 密钥和所选模型
bot = chatgpt(api_key="{YOUR_API_KEY}", engine="gpt-4o")

# 获取回答
result = bot.ask("python list use")

# 发送请求并实时获取流式响应
for text in bot.ask_stream("python list use"):
    print(text, end="")

# 关闭所有插件
bot = chatgpt(api_key="{YOUR_API_KEY}", engine="gpt-4o", use_plugins=False)
```

## 🍃 环境变量

以下是跟插件设置相关的环境变量列表：

| 变量名称 | 描述 | 必需的？ |
|---------------|-------------|-----------|
| SEARCH | 是否启用搜索插件。默认值为 `True`。 | 否 |
| URL | 是否启用URL摘要插件。默认值为 `True`。 | 否 |
| ARXIV | 是否启用arXiv论文摘要插件。默认值为 `False`。 | 否 |
| CODE | 是否启用代码解释器插件。默认值为 `False`。 | 否 |
| IMAGE | 是否启用图像生成插件。默认值为 `False`。 | 否 |
| DATE | 是否启用日期插件。默认值为 `False`。 | 否 |

## 支持的模型

- GPT-3.5/4/4 Turbo/4o
- o1-preview/o1-mini
- DALL-E 3
- Claude2/3/3.5
- Gemini1.5 Pro/Flash
- Vertex AI(Claude, Gemini)
- Groq
- DuckDuckGo(gpt-4o-mini, claude-3-haiku, Meta-Llama-3.1-70B, Mixtral-8x7B)

## 🧩 插件

本项目支持多种插件，包括：DuckDuckGo 和 Google 搜索、URL 摘要、ArXiv 论文摘要、DALLE-3 画图和代码解释器等。您可以通过设置环境变量来启用或禁用这些插件。

- 如何开发插件？

插件相关的代码全部在本仓库 git 子模块 ModelMerge 里面，ModelMerge 是我开发的一个独立的仓库，用于处理 API 请求，对话历史记录管理等功能。当你使用 git clone 的 --recurse-submodules 参数克隆本仓库后，ModelMerge 会自动下载到本地。插件所有的代码在本仓库中的相对路径为 `ModelMerge/src/ModelMerge/plugins`。你可以在这个目录下添加自己的插件代码。插件开发的流程如下：

1. 在 `ModelMerge/src/ModelMerge/plugins` 目录下创建一个新的 Python 文件，例如 `myplugin.py`。在 `ModelMerge/src/ModelMerge/plugins/__init__.py` 文件中导入你的插件，例如 `from .myplugin import MyPlugin`，在 `__all__` 列表中添加你的插件名称。

2. 在 `ModelMerge/src/ModelMerge/plugins/config.py` 里面的 `function_call_list` 变量中添加你的插件的OpenAI格式的tool请求体转换代码，只需要添加一行代码即可实现转换。Claude Gemini tool 请求体不需要额外编写，程序在请求 Gemini 或者 Claude API 的时候，会自动转换为 Claude/Gemini tool 格式。`function_call_list` 是一个字典，键是插件的名称，值是插件的请求体。请保证`function_call_list` 字典的键名保证唯一性，不能和已有的插件键名重复。

3. 在 `ModelMerge/src/ModelMerge/plugins/config.py` 里面的 `PLUGINS` 字典里面添加键值对，键是插件的名称，值是插件的环境变量及其默认值。这个默认值是插件的开关，如果默认值是`True`，那么插件默认是开启的，如果默认值是 `False`，那么插件默认是关闭的，需要在用户在 `/info` 命令里面手动开启。

4. 最后，在 `ModelMerge/src/ModelMerge/plugins/config.py` 里面的函数 `get_tools_result_async` 添加插件调用的代码，当机器人需要调用插件的时候，会调用这个函数。你需要在这个函数里面添加插件的调用代码。

完成上面的步骤，你的插件就可以使用了。🎉

## 许可证

本项目采用 MIT 许可证授权。

## 贡献

欢迎通过 GitHub 提交问题或拉取请求来贡献改进。

## 联系方式

如有任何疑问或需要帮助，请通过 [yym68686@outlook.com](mailto:yym68686@outlook.com) 联系我们。