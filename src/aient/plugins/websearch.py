import os
import re
import datetime
import requests
import threading
import time as record_time
from itertools import islice
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from .registry import register_tool

class ThreadWithReturnValue(threading.Thread):
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return

import re
import httpx
import lxml.html
from lxml_html_clean import Cleaner
from html2text import HTML2Text
from textwrap import dedent

def url_to_markdown(url):
    # 获取并清理网页内容
    def get_body(url):
        try:
            text = httpx.get(url, verify=False, timeout=5).text
            if text == "":
                return "抱歉，目前无法访问该网页。"
            # body = lxml.html.fromstring(text).xpath('//body')

            doc = lxml.html.fromstring(text)
            # 检查是否是GitHub raw文件格式（body > pre）
            if doc.xpath('//body/pre'):
                return text  # 直接返回原始文本，保留格式

            body = doc.xpath('//body')
            if body == [] and text != "":
                body = text
                return f'<pre>{body}</pre>'
                # return body
            else:
                body = body[0]
                body = Cleaner(javascript=True, style=True).clean_html(body)
                return ''.join(lxml.html.tostring(c, encoding='unicode') for c in body)
        except Exception as e:
            # print('\033[31m')
            # print("error: url_to_markdown url", url)
            # print("error", e)
            # print('\033[0m')
            return "抱歉，目前无法访问该网页。"

    # 将HTML转换为Markdown
    def get_md(cts):
        h2t = HTML2Text(bodywidth=5000)
        h2t.ignore_links = True
        h2t.mark_code = True
        h2t.ignore_images = True
        res = h2t.handle(cts)

        def _f(m):
            return f'```\n{dedent(m.group(1))}\n```'

        return re.sub(r'\[code]\s*\n(.*?)\n\[/code]', _f, res or '', flags=re.DOTALL).strip()

    # 获取网页内容
    body_content = get_body(url)

    # 转换为Markdown
    markdown_content = get_md(body_content)

    return "URL Source: " + url + "\n\ntext: " + markdown_content

def jina_ai_Web_crawler(url: str, isSearch=False) -> str:
    """返回链接网址url正文内容，必须是合法的网址"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    result = ''
    try:
        requests.packages.urllib3.disable_warnings()
        url = "https://r.jina.ai/" + url
        response = requests.get(url, headers=headers, verify=False, timeout=5, stream=True)
        if response.status_code == 404:
            print("Page not found:", url)
            return "抱歉，网页不存在，目前无法访问该网页。@Trash@"
        content_length = int(response.headers.get('Content-Length', 0))
        if content_length > 5000000:
            print("Skipping large file:", url)
            return result

        # 检查内容是否为HTML
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type or 'application/xhtml+xml' in content_type:
            # 使用html.parser而不是lxml可能会更宽松一些
            soup = BeautifulSoup(response.content, 'html.parser')
        else:
            # 对于非HTML内容，直接提取文本
            return response.text  # 限制长度

        table_contents = ""
        tables = soup.find_all('table')
        for table in tables:
            table_contents += table.get_text()
            table.decompose()
        body = "".join(soup.find('body').get_text().split('\n'))
        result = table_contents + body
        if result == '' and not isSearch:
            result = "抱歉，可能反爬虫策略，目前无法访问该网页。@Trash@"
        if result.count("\"") > 1000:
            result = ""
    except Exception as e:
        # print('\033[31m')
        # print("error: jina_ai_Web_crawler url", url)
        # print("error", e)
        # print('\033[0m')
        pass
    # print(result + "\n\n")
    return result

@register_tool()
def get_url_content(url: str) -> str:
    """
    获取 url 的网页内容，以 markdown 格式返回给用户

    :param url: 要爬取的网页URL
    :return: 网页内容
    """
    markdown_content = url_to_markdown(url)
    # print(markdown_content)
    # print('-----------------------------')
    jina_content = jina_ai_Web_crawler(url)
    # print('-----------------------------')

    # 定义评分函数
    def score_content(content):
        # 1. 内容长度
        length_score = len(content)

        # 2. 是否包含错误信息
        error_penalty = 1000 if "抱歉" in content or "@Trash@" in content else 0

        # 3. 内容的多样性（可以通过不同类型的字符来粗略估计）
        diversity_score = len(set(content))

        # 4. 特殊字符比例（过高可能意味着格式问题）
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', content)) / len(content)
        special_char_penalty = 500 if special_char_ratio > 0.1 else 0

        return length_score + diversity_score - error_penalty - special_char_penalty

    if markdown_content == "":
        markdown_score = -2000
    else:
        markdown_score = score_content(markdown_content)
    if jina_content == "":
        jina_score = -2000
    else:
        jina_score = score_content(jina_content)

    # print(f"url_to_markdown 得分： {markdown_score}")
    # print(f"jina_ai_Web_crawler 得分： {jina_score}")

    if markdown_score > jina_score:
        # print("choose: 选择 url_to_markdown 的结果")
        return markdown_content
    elif markdown_score == jina_score and jina_score < 0:
        print("choose: 两者都无法访问")
        return ""
    else:
        # print("choose: 选择 jina_ai_Web_crawler 的结果")
        return jina_content

def getddgsearchurl(query, max_results=4):
    try:
        results = []
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(query, safesearch='Off', timelimit='y', backend="lite")
            for r in islice(ddgs_gen, max_results):
                results.append(r)
        urls = [result['href'] for result in results]
    except Exception as e:
        print('\033[31m')
        print("duckduckgo error", e)
        print('\033[0m')
        urls = []
    return urls

def getgooglesearchurl(result, numresults=3):
    urls = []
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': result,
            'key': os.environ.get('GOOGLE_API_KEY', None),
            'cx': os.environ.get('GOOGLE_CSE_ID', None)
        }
        response = requests.get(url, params=params)
        # print(response.text)
        results = response.json()
        link_list = [item['link'] for item in results.get('items', [])]
        urls = link_list[:numresults]
    except Exception as e:
        print('\033[31m')
        print("error", e)
        print('\033[0m')
        if "rateLimitExceeded" in str(e):
            print("Google API 每日调用频率已达上限，请明日再试！")
    # print("google urls", urls)
    return urls

def sort_by_time(urls):
    def extract_date(url):
        match = re.search(r'[12]\d{3}.\d{1,2}.\d{1,2}', url)
        if match is not None:
            match = re.sub(r'([12]\d{3}).(\d{1,2}).(\d{1,2})', "\\1/\\2/\\3", match.group())
            print(match)
            if int(match[:4]) > datetime.datetime.now().year:
                match = "1000/01/01"
        else:
            match = "1000/01/01"
        try:
            return datetime.datetime.strptime(match, '%Y/%m/%d')
        except:
            match = "1000/01/01"
            return datetime.datetime.strptime(match, '%Y/%m/%d')

    # 提取日期并创建一个包含日期和URL的元组列表
    date_url_pairs = [(extract_date(url), url) for url in urls]

    # 按日期排序
    date_url_pairs.sort(key=lambda x: x[0], reverse=True)

    # 获取排序后的URL列表
    sorted_urls = [url for _, url in date_url_pairs]

    return sorted_urls

async def get_search_url(keywords, search_url_num):
    yield "message_search_stage_2"

    search_threads = []
    if os.environ.get('GOOGLE_API_KEY', None) and os.environ.get('GOOGLE_CSE_ID', None):
        search_thread = ThreadWithReturnValue(target=getgooglesearchurl, args=(keywords[0],search_url_num,))
        keywords.pop(0)
        search_thread.start()
        search_threads.append(search_thread)

    urls_set = []
    urls_set += getddgsearchurl(keywords[0], search_url_num)

    for t in search_threads:
        tmp = t.join()
        urls_set += tmp
    url_set_list = sorted(set(urls_set), key=lambda x: urls_set.index(x))
    url_set_list = sort_by_time(url_set_list)

    url_pdf_set_list = [item for item in url_set_list if item.endswith(".pdf")]
    url_set_list = [item for item in url_set_list if not item.endswith(".pdf")]
    # cut_num = int(len(url_set_list) * 1 / 3)
    yield url_set_list[:6], url_pdf_set_list
    # return url_set_list[:6], url_pdf_set_list
    # return url_set_list, url_pdf_set_list

def concat_url(threads):
    url_result = []
    for t in threads:
        tmp = t.join()
        if tmp:
            url_result.append(tmp)
    return url_result

async def get_url_text_list(keywords, search_url_num):
    start_time = record_time.time()

    async for chunk in get_search_url(keywords, search_url_num):
        if type(chunk) == str:
            yield chunk
        else:
            url_set_list, url_pdf_set_list = chunk
    # url_set_list, url_pdf_set_list = yield from get_search_url(keywords, search_url_num)

    yield "message_search_stage_3"
    threads = []
    for url in url_set_list:
        # url_search_thread = ThreadWithReturnValue(target=jina_ai_Web_crawler, args=(url,True,))
        url_search_thread = ThreadWithReturnValue(target=get_url_content, args=(url,))
        # url_search_thread = ThreadWithReturnValue(target=Web_crawler, args=(url,True,))
        url_search_thread.start()
        threads.append(url_search_thread)

    url_text_list = concat_url(threads)

    yield "message_search_stage_4"
    end_time = record_time.time()
    run_time = end_time - start_time
    print("urls", url_set_list)
    print(f"搜索用时：{run_time}秒")

    yield url_text_list
    # return url_text_list

# Plugins 搜索入口
@register_tool()
async def get_search_results(query):
    """
    执行网络搜索并返回搜索结果文本

    参数:
        query: 查询语句，包含用户想要搜索的内容

    返回:
        异步生成器，依次产生:
        - 搜索状态消息 ("message_search_stage_2", "message_search_stage_3", "message_search_stage_4")
        - 最终的搜索结果文本列表

    说明:
        - 根据查询语句自动搜索结果
        - 使用多线程并行抓取网页内容
        - 在搜索过程中通过yield返回状态更新
    """
    keywords = query
    if len(keywords) == 3:
        search_url_num = 4
    if len(keywords) == 2:
        search_url_num = 6
    if len(keywords) == 1:
        search_url_num = 12

    url_text_list = []
    async for chunk in get_url_text_list(keywords, search_url_num):
        if type(chunk) == str:
            yield chunk
        else:
            url_text_list = chunk
    yield url_text_list

if __name__ == "__main__":
    os.system("clear")
    # from aient.models import chatgpt
    # print(get_search_results("今天的微博热搜有哪些？", chatgpt.chatgpt_api_url.v1_url))

    # # 搜索

    # for i in search_web_and_summary("今天的微博热搜有哪些？"):
    # for i in search_web_and_summary("给出清华铊中毒案时间线，并作出你的评论。"):
    # for i in search_web_and_summary("红警hbk08是谁"):
    # for i in search_web_and_summary("国务院 2024 放假安排"):
    # for i in search_web_and_summary("中国最新公布的游戏政策，对游戏行业和其他相关行业有什么样的影响？"):
    # for i in search_web_and_summary("今天上海的天气怎么样？"):
    # for i in search_web_and_summary("阿里云24核96G的云主机价格是多少"):
    # for i in search_web_and_summary("话说葬送的芙莉莲动漫是半年番还是季番？完结没？"):
    # for i in search_web_and_summary("周海媚事件进展"):
    # for i in search_web_and_summary("macos 13.6 有什么新功能"):
    # for i in search_web_and_summary("用python写个网络爬虫给我"):
    # for i in search_web_and_summary("消失的她主要讲了什么？"):
    # for i in search_web_and_summary("奥巴马的全名是什么？"):
    # for i in search_web_and_summary("华为mate60怎么样？"):
    # for i in search_web_and_summary("慈禧养的猫叫什么名字?"):
    # for i in search_web_and_summary("民进党当初为什么支持柯文哲选台北市长？"):
    # for i in search_web_and_summary("Has the United States won the china US trade war？"):
    # for i in search_web_and_summary("What does 'n+2' mean in Huawei's 'Mate 60 Pro' chipset? Please conduct in-depth analysis."):
    # for i in search_web_and_summary("AUTOMATIC1111 是什么？"):
    # for i in search_web_and_summary("python telegram bot 怎么接收pdf文件"):
    # for i in search_web_and_summary("中国利用外资指标下降了 87% ？真的假的。"):
    # for i in search_web_and_summary("How much does the 'zeabur' software service cost per month? Is it free to use? Any limitations?"):
    # for i in search_web_and_summary("英国脱欧没有好处，为什么英国人还是要脱欧？"):
    # for i in search_web_and_summary("2022年俄乌战争为什么发生？"):
    # for i in search_web_and_summary("卡罗尔与星期二讲的啥？"):
    # for i in search_web_and_summary("金砖国家会议有哪些决定？"):
    # for i in search_web_and_summary("iphone15有哪些新功能？"):
    # for i in search_web_and_summary("python函数开头：def time(text: str) -> str:每个部分有什么用？"):
        # print(i, end="")

    # 问答
    # result = asyncio.run(docQA("/Users/yanyuming/Downloads/GitHub/wiki/docs", "ubuntu 版本号怎么看？"))
    # result = asyncio.run(docQA("https://yym68686.top", "说一下HSTL pipeline"))
    # result = asyncio.run(docQA("https://wiki.yym68686.top", "PyTorch to MindSpore翻译思路是什么？"))
    # print(result['answer'])
    # result = asyncio.run(pdfQA("https://api.telegram.org/file/bot5569497961:AAHobhUuydAwD8SPkXZiVFybvZJOmGrST_w/documents/file_1.pdf", "HSTL的pipeline详细讲一下"))
    # print(result)
    # source_url = set([i.metadata['source'] for i in result["source_documents"]])
    # source_url = "\n".join(source_url)
    # message = (
    #     f"{result['result']}\n\n"
    #     f"参考链接：\n"
    #     f"{source_url}"
    # )
    # print(message)