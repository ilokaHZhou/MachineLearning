# from langchain_ollama import OllamaEmbeddings

# embeddings = OllamaEmbeddings(model="EntropyYue/jina-embeddings-v2-base-zh")

# vec = embeddings.embed_query("你好，测试一下 embedding")
# print(len(vec))  # 应该是 768 维向量

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
vec = emb.embed_query("hello world")

print("Embedding dimension:", len(vec)) # 应该是3072维向量
