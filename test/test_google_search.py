import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
load_dotenv()

search_engine_id = os.environ.get('GOOGLE_CSE_ID', None)
api_key = os.environ.get('GOOGLE_API_KEY', None)

def google_search(query, api_key, search_engine_id):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=search_engine_id).execute()
    link_list = [item['link'] for item in res['items']]
    return link_list

# 执行搜索
query = "Python programming"
results = google_search(query, api_key, search_engine_id)
print(results)

# # 处理搜索结果
# for result in results:
#     print(result['title'])
#     print(result['link'])
#     print(result['snippet'])
#     print("---")