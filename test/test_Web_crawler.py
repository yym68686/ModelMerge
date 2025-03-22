import re
import os
os.system('cls' if os.name == 'nt' else 'clear')
import time
import requests
from bs4 import BeautifulSoup

def Web_crawler(url: str, isSearch=False) -> str:
    """返回链接网址url正文内容，必须是合法的网址"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    result = ''
    try:
        requests.packages.urllib3.disable_warnings()
        response = requests.get(url, headers=headers, verify=False, timeout=3, stream=True)
        if response.status_code == 404:
            print("Page not found:", url)
            return ""
            # return "抱歉，网页不存在，目前无法访问该网页。@Trash@"
        content_length = int(response.headers.get('Content-Length', 0))
        if content_length > 5000000:
            print("Skipping large file:", url)
            return result
        try:
            soup = BeautifulSoup(response.text.encode(response.encoding), 'xml', from_encoding='utf-8')
        except:
            soup = BeautifulSoup(response.text.encode(response.encoding), 'html.parser', from_encoding='utf-8')
        # print("soup", soup)

        for script in soup(["script", "style"]):
            script.decompose()

        table_contents = ""
        tables = soup.find_all('table')
        for table in tables:
            table_contents += table.get_text()
            table.decompose()

        # body_text = "".join(soup.find('body').get_text().split('\n'))
        body = soup.find('body')
        if body:
            body_text = body.get_text(separator=' ', strip=True)
        else:
            body_text = soup.get_text(separator=' ', strip=True)

        result = table_contents + body_text
        if result == '' and not isSearch:
            result = ""
            # result = "抱歉，可能反爬虫策略，目前无法访问该网页。@Trash@"
        if result.count("\"") > 1000:
            result = ""
    except Exception as e:
        print('\033[31m')
        print("error: url", url)
        print("error", e)
        print('\033[0m')
        result = "抱歉，目前无法访问该网页。"
    # print("url content", result + "\n\n")
    print(result)
    return result

import lxml.html
from lxml.html.clean import Cleaner
import httpx
def get_body(url):
    body = lxml.html.fromstring(httpx.get(url).text).xpath('//body')[0]
    body = Cleaner(javascript=True, style=True).clean_html(body)
    return ''.join(lxml.html.tostring(c, encoding='unicode') for c in body)

import re
import httpx
import lxml.html
from lxml.html.clean import Cleaner
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
            print('\033[31m')
            print("error: url", url)
            print("error", e)
            print('\033[0m')
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

    return markdown_content

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
        soup = BeautifulSoup(response.text.encode(response.encoding), 'lxml', from_encoding='utf-8')
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
        print('\033[31m')
        print("error: url", url)
        print("error", e)
        print('\033[0m')
        result = "抱歉，目前无法访问该网页。"
    print(result + "\n\n")
    return result


def get_url_content(url: str) -> str:
    """
    比较 url_to_markdown 和 jina_ai_Web_crawler 的结果，选择更好的内容

    :param url: 要爬取的网页URL
    :return: 选择的更好的内容
    """
    markdown_content = url_to_markdown(url)
    print(markdown_content)
    print('-----------------------------')
    jina_content = jina_ai_Web_crawler(url)
    print('-----------------------------')

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

    print(f"url_to_markdown 得分： {markdown_score}")
    print(f"jina_ai_Web_crawler 得分： {jina_score}")

    if markdown_score > jina_score:
        print("选择 url_to_markdown 的结果")
        return markdown_content
    elif markdown_score == jina_score and jina_score < 0:
        print("两者都无法访问")
        return ""
    else:
        print("选择 jina_ai_Web_crawler 的结果")
        return jina_content

start_time = time.time()
# for url in ['https://www.zhihu.com/question/557257320', 'https://job.achi.idv.tw/2021/12/05/what-is-the-403-forbidden-error-how-to-fix-it-8-methods-explained/', 'https://www.lifewire.com/403-forbidden-error-explained-2617989']:
# for url in ['https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/403']:
# for url in ['https://www.hostinger.com/tutorials/what-is-403-forbidden-error-and-how-to-fix-it']:
# for url in ['https://beebom.com/what-is-403-forbidden-error-how-to-fix/']:
# for url in ['https://www.lifewire.com/403-forbidden-error-explained-2617989']:
# for url in ['https://www.usnews.com/news/best-countries/articles/2022-02-24/explainer-why-did-russia-invade-ukraine']:
# for url in ['https://github.com/EAimTY/tuic']:
# TODO 没办法访问
# for url in ['https://s.weibo.com/top/summary?cate=realtimehot']:
# for url in ['https://www.microsoft.com/en-us/security/blog/2023/05/24/volt-typhoon-targets-us-critical-infrastructure-with-living-off-the-land-techniques/']:
# for url in ['https://tophub.today/n/KqndgxeLl9']:
# for url in ['https://support.apple.com/zh-cn/HT213931']:
# for url in ["https://zeta.zeabur.app"]:
# for url in ["https://www.anthropic.com/research/probes-catch-sleeper-agents"]:
# for url in ['https://finance.sina.com.cn/stock/roll/2023-06-26/doc-imyyrexk4053724.shtml']:
# for url in ['https://s.weibo.com/top/summary?cate=realtimehot']:
# for url in ['https://tophub.today/n/KqndgxeLl9', 'https://www.whatsonweibo.com/', 'https://www.trendingonweibo.com/?ref=producthunt', 'https://www.trendingonweibo.com/', 'https://www.statista.com/statistics/1377073/china-most-popular-news-on-weibo/']:
# for url in ['https://www.usnews.com/news/entertainment/articles/2023-12-22/china-drafts-new-rules-proposing-restrictions-on-online-gaming']:
# for url in ['https://developer.aliyun.com/article/721836']:
# for url in ['https://cn.aliyun.com/page-source/price/detail/machinelearning_price']:
# for url in ['https://mp.weixin.qq.com/s/Itad7Y-QBcr991JkF3SrIg']:
# for url in ['https://zhidao.baidu.com/question/317577832.html']:
# for url in ['https://www.cnn.com/2023/09/06/tech/huawei-mate-60-pro-phone/index.html']:
# for url in ['https://www.reddit.com/r/China_irl/comments/15qojkh/46%E6%9C%88%E5%A4%96%E8%B5%84%E5%AF%B9%E4%B8%AD%E5%9B%BD%E7%9B%B4%E6%8E%A5%E6%8A%95%E8%B5%84%E5%87%8F87/', 'https://www.apple.com.cn/job-creation/Apple_China_CSR_Report_2020.pdf', 'https://hdr.undp.org/system/files/documents/hdr2013chpdf.pdf']:
# for url in ['https://www.airuniversity.af.edu/JIPA/Display/Article/3111127/the-uschina-trade-war-vietnam-emerges-as-the-greatest-winner/']:
# for url in ['https://zhuanlan.zhihu.com/p/646786536']:
# for url in ['https://zh.wikipedia.org/wiki/%E4%BF%84%E7%BE%85%E6%96%AF%E5%85%A5%E4%BE%B5%E7%83%8F%E5%85%8B%E8%98%AD']:
for url in ['https://raw.githubusercontent.com/yym68686/ChatGPT-Telegram-Bot/main/README.md']:
# for url in ['https://raw.githubusercontent.com/openai/openai-python/main/src/openai/api_requestor.py']:
# for url in ['https://stock.finance.sina.com.cn/usstock/quotes/aapl.html']:
    # Web_crawler(url)
    # print(get_body(url))
    # print('-----------------------------')
    # jina_ai_Web_crawler(url)
    # print('-----------------------------')
    # print(url_to_markdown(url))
    # print('-----------------------------')
    best_content = get_url_content(url)
end_time = time.time()
run_time = end_time - start_time
# 打印运行时间
print(f"程序运行时间：{run_time}秒")
