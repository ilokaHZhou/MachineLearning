from dotenv import load_dotenv
import os
from pprint import pprint

from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import DeepSeekEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_ollama import OllamaEmbeddings

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from splitter import split_docs_by_markdown_headers
from loader import load_docs

def embedding():
    load_dotenv()

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    print("PINECONE_API_KEY:", pinecone_api_key)
    pc = Pinecone(api_key=pinecone_api_key)

    # 创建 Pinecone index（如果不存在）
    # index_name = "my-md-rag-index"
    # if index_name not in pc.list_indexes().names():
    #     pc.create_index(
    #         name=index_name,
    #         dimension=1536, 
    #         metric="cosine",
    #         spec=ServerlessSpec(
    #             cloud="aws",
    #             region="us-east-1"
    #         )
    #     )
    index_name = "dochelper"
    index = pc.Index(index_name)

    # 1. 加载文档
    root = "D:/GitHub_Repos/InterviewAndLeetCode/面试题"
    docs = load_docs(root)

    # 2. 按 Markdown 标题层级切分文档
    chunks = split_docs_by_markdown_headers(docs)
    print(f"✅ 按标题切分完成，共 {len(chunks)} 个块")

    # 确定一下每个chunk里原文内容的字段叫什么，后面embedding和向量库要用到
    # pprint(chunks[0].dict())
    # print(type(chunks[0]))

    # 3. 创建向量embedding模型

    # 只能连了openai的api key才能直接用
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Deepseek查不到确切的embedding模型和向量维度
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding",                # DeepSeek 的 embedding 模型名
    #     api_key=deepseek_api_key,
    #     base_url="https://api.deepseek.com"    # 必须设置为 DeepSeek 域名！
    # )


    """
    改用HuggingFaces上的 千问 embedding模型

    千问7B版本内存占用太大，要30个G了
    model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"

    1.5B版本相对小很多，4-8G显存就能跑但是使用会现下载，也很慢，所以下面使用事先下好的本地模型路径
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

    设定国内镜像然后下载模型到本地的命令：
    pip install huggingface_hub
    set HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download Alibaba-NLP/gte-Qwen2-1.5B-instruct --local-dir ./gte1_5b --local-dir-use-symlinks False
    """

    # embeddings = HuggingFaceEmbeddings(
    #     model_name="./gte1_5b",
    #     model_kwargs={
    #         "device": "cpu", # 如果有GPU, 可以加速，但是Windows原生环境装不了 flash_attn，CUDA会报错
    #         "trust_remote_code": True # 千问模型里有自定义代码，需要加这个参数授权才能没有报错
    #     },
    #     encode_kwargs={"normalize_embeddings": True}  # 推荐归一化，把每个embedding向量拉到相同长度，否则检索时向量长的得分会偏高
    # )

    """
    改用 Ollama 的中文 embedding 模型
    """
    embeddings = OllamaEmbeddings(model="EntropyYue/jina-embeddings-v2-base-zh")

    # 测试向量维度
    # vec = embeddings.embed_query("你好")
    # print("向量维度:", len(vec))

    # 3. 写入向量数据库
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="page_content"
    )

    vectorstore.add_documents(chunks, batch_size=64) # 分批写入，避免一次性数据量过大太慢

    print("🎉 Embedding + Pinecone ingest 完成！")



if __name__ == "__main__":
    embedding()