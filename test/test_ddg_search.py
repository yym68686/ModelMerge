from itertools import islice
from duckduckgo_search import DDGS

# def getddgsearchurl(query, max_results=4):
#     try:
#         webresult = DDGS().text(query, max_results=max_results)
#         if webresult == None:
#             return []
#         urls = [result['href'] for result in webresult]
#     except Exception as e:
#         print('\033[31m')
#         print("duckduckgo error", e)
#         print('\033[0m')
#         urls = []
#     # print("ddg urls", urls)
#     return urls

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

def search_answers(keywords, max_results=4):
    results = []
    with DDGS() as ddgs:
        # 使用DuckDuckGo搜索关键词
        ddgs_gen = ddgs.answers(keywords)
        # 从搜索结果中获取最大结果数
        for r in islice(ddgs_gen, max_results):
            results.append(r)

    # 返回一个json响应，包含搜索结果
    return {'results': results}


if __name__ == '__main__':
    # 搜索关键词
    query = "OpenAI"
    print(getddgsearchurl(query))
    # print(search_answers(query))