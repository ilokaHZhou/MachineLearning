# Please install OpenAI SDK first: `pip3 install openai`

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

deepseek_api_url = os.getenv("DEEPSEEK_API_URL")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)