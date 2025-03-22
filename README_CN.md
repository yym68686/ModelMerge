# aient

[英文](./README.md) | [中文](./README_CN.md)

aient 是一个强大的库，旨在简化和统一不同大型语言模型的使用，包括 GPT-3.5/4/4 Turbo/4o、o1-preview/o1-mini、DALL-E 3、Claude2/3/3.5、Gemini1.5 Pro/Flash、Vertex AI(Claude, Gemini) 、DuckDuckGo 和 Groq。该库支持 GPT 格式的函数调用，并内置了 Google 搜索和 URL 总结功能，极大地增强了模型的实用性和灵活性。

## ✨ 特性

- **多模型支持**：集成多种最新的大语言模型。
- **实时交互**：支持实时查询流，实时获取模型响应。
- **功能扩展**：通过内置的函数调用（function calling）支持，可以轻松扩展模型的功能，目前支持 DuckDuckGo 和 Google 搜索、内容摘要、Dalle-3画图、arXiv 论文总结、当前时间、代码解释器等插件。
- **简易接口**：提供简洁统一的 API 接口，使得调用和管理模型变得轻松。

## 快速上手

以下是如何在您的 Python 项目中快速集成和使用 aient 的指南。

### 安装

首先，您需要安装 aient。可以通过 pip 直接安装：

```bash
pip install aient
```

### 使用示例

以下是一个简单的示例，展示如何使用 aient 来请求 GPT-4 模型并处理返回的流式数据：

```python
from aient import chatgpt

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
| get_search_results | 是否启用搜索插件。默认值为 `False`。 | 否 |
| get_url_content | 是否启用URL摘要插件。默认值为 `False`。 | 否 |
| download_read_arxiv_pdf | 是否启用arXiv论文摘要插件。默认值为 `False`。 | 否 |
| run_python_script | 是否启用代码解释器插件。默认值为 `False`。 | 否 |
| generate_image | 是否启用图像生成插件。默认值为 `False`。 | 否 |
| get_date_time_weekday | 是否启用日期插件。默认值为 `False`。 | 否 |

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

插件相关的代码全部在本仓库 git 子模块 aient 里面，aient 是我开发的一个独立的仓库，用于处理 API 请求，对话历史记录管理等功能。当你使用 git clone 的 --recurse-submodules 参数克隆本仓库后，aient 会自动下载到本地。插件所有的代码在本仓库中的相对路径为 `aient/src/aient/plugins`。你可以在这个目录下添加自己的插件代码。插件开发的流程如下：

1. 在 `aient/src/aient/plugins` 目录下创建一个新的 Python 文件，例如 `myplugin.py`。通过在函数上面添加 `@register_tool()` 装饰器注册插件。`register_tool` 通过 `from .registry import register_tool` 导入。

完成上面的步骤，你的插件就可以使用了。🎉

## 许可证

本项目采用 MIT 许可证授权。

## 贡献

欢迎通过 GitHub 提交问题或拉取请求来贡献改进。

## 联系方式

如有任何疑问或需要帮助，请通过 [yym68686@outlook.com](mailto:yym68686@outlook.com) 联系我们。