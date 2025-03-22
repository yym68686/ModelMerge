function_call_list = \
{
    "base": {
        "tools": [],
        "tool_choice": "auto"
    },
    "current_weather": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ]
                }
            },
            "required": [
                "location"
            ]
        }
    },
    "SEARCH": {
        "name": "get_search_results",
        "description": "Search Google to enhance knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to search."
                }
            },
            "required": [
                "prompt"
            ]
        }
    },
    "URL": {
        "name": "get_url_content",
        "description": "Get the webpage content of a URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "the URL to request"
                }
            },
            "required": [
                "url"
            ]
        }
    },
    "DATE": {
        "name": "get_date_time_weekday",
        "description": "Get the current time, date, and day of the week",
    },
    "VERSION": {
        "name": "get_version_info",
        "description": "Get version information",
    },
    "TARVEL": {
        "name": "get_city_tarvel_info",
        "description": "Get the city's travel plan by city name.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "the city to search"
                }
            },
            "required": [
                "city"
            ]
        }
    },
    "IMAGE": {
        "name": "generate_image",
        "description": "Generate images based on user descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "the prompt to generate image"
                }
            },
            "required": [
                "prompt"
            ]
        }
    },
    "CODE": {
        "name": "run_python_script",
        "description": "Convert the string to a Python script and return the Python execution result. Assign the result to the variable result. The results must be printed to the console using the print function. Directly output the code, without using quotation marks or other symbols to enclose the code.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "the code to run"
                }
            },
            "required": [
                "prompt"
            ]
        }
    },
    "ARXIV": {
        "name": "download_read_arxiv_pdf",
        "description": "Get the content of the paper corresponding to the arXiv ID",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "the arXiv ID of the paper"
                }
            },
            "required": [
                "prompt"
            ]
        }
    },
    "FLIGHT": {
        "name": "get_Round_trip_flight_price",
        "description": "Get round-trip ticket prices between two cities for the next six months. Use two city names as parameters. The name of the citys must be in Chinese.",
        "parameters": {
            "type": "object",
            "properties": {
                "departcity": {
                    "type": "string",
                    "description": "the chinese name of departure city. e.g. 上海"
                },
                "arrivalcity": {
                    "type": "string",
                    "description": "the chinese name of arrival city. e.g. 北京"
                }
            },
            "required": [
                "departcity",
                "arrivalcity"
            ]
        }
    },
}


if __name__ == "__main__":
    import json
    tools_list = {"tools": [{"type": "function", "function": function_call_list[key]} for key in function_call_list.keys() if key != "base"]}
    print(json.dumps(tools_list, indent=4, ensure_ascii=False))