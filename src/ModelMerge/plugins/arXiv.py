import requests

from ..utils.scripts import Document_extract
from .registry import register_tool

@register_tool()
async def download_read_arxiv_pdf(arxiv_id: str) -> str:
    """
    下载指定arXiv ID的论文PDF并提取其内容。

    此函数会下载arXiv上的论文PDF文件，保存到指定路径，
    然后使用文档提取工具读取其内容。

    Args:
        arxiv_id: arXiv论文的ID，例如'2305.12345'

    Returns:
        提取的论文内容文本或失败消息
    """
    # 构造下载PDF的URL
    url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

    # 发送HTTP GET请求
    response = requests.get(url)

    # 检查是否成功获取内容
    if response.status_code == 200:
        # 将PDF内容写入文件
        save_path = "paper.pdf"
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f'PDF下载成功，保存路径: {save_path}')
        return await Document_extract(None, save_path)
    else:
        print(f'下载失败，状态码: {response.status_code}')
        return "文件下载失败"

if __name__ == '__main__':
    # 示例使用
    arxiv_id = '2305.12345'  # 替换为实际的arXiv ID

    # 测试下载功能
    # print(download_read_arxiv_pdf(arxiv_id))

    # 测试函数转换为JSON
    # json_result = function_to_json(download_read_arxiv_pdf)
    # import json
    # print(json.dumps(json_result, indent=2, ensure_ascii=False))