import os
import requests
from googleapiclient.discovery import build
from dotenv import load_dotenv
load_dotenv()

search_engine_id = os.environ.get('GOOGLE_CSE_ID', None)
api_key = os.environ.get('GOOGLE_API_KEY', None)
query = "Python 编程"

def google_search1(query, api_key, search_engine_id):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=search_engine_id).execute()
    link_list = [item['link'] for item in res['items']]
    return link_list

def google_search2(query, api_key, cx):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cx
    }
    response = requests.get(url, params=params)
    print(response.text)
    results = response.json()
    link_list = [item['link'] for item in results.get('items', [])]

    return link_list

# results = google_search1(query, api_key, search_engine_id)
# print(results)

results = google_search2(query, api_key, search_engine_id)
print(results)