import json

from ..src.aient.plugins.websearch import get_search_results
from ..src.aient.plugins.arXiv import download_read_arxiv_pdf
from ..src.aient.plugins.image import generate_image
from ..src.aient.plugins.today import get_date_time_weekday
from ..src.aient.plugins.run_python import run_python_script

from ..src.aient.plugins.config import function_to_json


print(json.dumps(function_to_json(get_search_results), indent=4, ensure_ascii=False))
print(json.dumps(function_to_json(download_read_arxiv_pdf), indent=4, ensure_ascii=False))
print(json.dumps(function_to_json(generate_image), indent=4, ensure_ascii=False))
print(json.dumps(function_to_json(get_date_time_weekday), indent=4, ensure_ascii=False))
print(json.dumps(function_to_json(run_python_script), indent=4, ensure_ascii=False))
