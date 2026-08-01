# hello_llm.py — first AI program
import os
from dotenv import load_dotenv
import openai

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit("OPENROUTER_API_KEY is missing. Add it to .env and try again.")

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="qwen/qwen3.5-flash-02-23",
    messages=[
        {"role": "user", "content": "What is Python in 2 sentences?"}
    ],
)

print(response.choices[0].message.content)
