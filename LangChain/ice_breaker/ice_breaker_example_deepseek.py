from langchain.llms import BaseLLM
from langchain_community.llms.utils import enforce_stop_tokens
import requests
from typing import Optional, List
from dotenv import load_dotenv
import os
from langchain.schema import Generation, LLMResult

# 加载.env文件中的环境变量
load_dotenv()

class DeepSeekLLM(BaseLLM):
    deepseek_api_url: str
    deepseek_api_key: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            'Authorization': f"Bearer {self.deepseek_api_key}",
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        data = {
            # 根据 DeepSeek API 的要求，设置其他必要的参数
            "messages": [
                # {
                # "content": "You are a helpful assistant",
                # "role": "system"
                # },
                {
                "content": prompt,
                "role": "user"
                }
            ],
            "model": "deepseek-chat",
            "frequency_penalty": 0,
            "max_tokens": 2048,
            "presence_penalty": 0,
            "response_format": {
                "type": "text"
            },
            "stop": None,
            "stream": False,
            "stream_options": None,
            "temperature": 1,
            "top_p": 1,
            "tools": None,
            "tool_choice": "none",
            "logprobs": False,

        }
        response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"DeepSeek API returned status code {response.status_code}: {response.text}")
        result = response.json()
        # 从 API 响应中提取生成的文本
        generated_text = result["choices"][0]["message"]["content"]
        if stop is not None:
            generated_text = enforce_stop_tokens(generated_text, stop)
        return generated_text

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            gen = Generation(text=text)
            generations.append([gen])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "deepseek"


if __name__ == "__main__":
    # 从环境变量中获取 API 端点和 API 密钥
    deepseek_api_url = os.getenv("DEEPSEEK_API_URL")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not deepseek_api_url or not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_URL or DEEPSEEK_API_KEY not found in environment variables")

    deepseek_llm = DeepSeekLLM(
        deepseek_api_url=deepseek_api_url,
        deepseek_api_key=deepseek_api_key
    )
    prompt = "请写一个简短的故事，17个字"
    response = deepseek_llm.invoke(prompt)
    print(response)
    