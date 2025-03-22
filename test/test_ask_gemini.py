import os
from aient.models import gemini

GOOGLE_AI_API_KEY = os.environ.get('GOOGLE_AI_API_KEY', None)

bot = gemini(api_key=GOOGLE_AI_API_KEY, engine='gemini-2.0-flash-exp')
for text in bot.ask_stream("give me some example code of next.js to build a modern web site"):
    print(text, end="")