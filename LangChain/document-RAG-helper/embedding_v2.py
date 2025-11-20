import os
from dotenv import load_dotenv
from splitter import split_docs_by_markdown_headers
from loader import load_docs
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def embedding():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "test"
    index = pc.Index(index_name)


    # 1. 加载文档
    root = "D:/GitHub_Repos/InterviewAndLeetCode/面试题"
    print("正在加载文档...")
    docs = load_docs(root)
    print("✅ 文档加载完成，共", len(docs), "个文档")

    # 2. 按 Markdown 标题层级切分文档
    print("正在按标题切分文档...")
    chunks = split_docs_by_markdown_headers(docs)
    print(f"✅ 按标题切分完成，共 {len(chunks)} 个块")
    print("示例 chunk：\n", chunks[7].page_content[:300])

    # 3. 创建向量embedding模型
    print("正在创建 Embedding 模型...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    print("✅ Embedding 模型创建完成")

    # 4. 写入向量数据库
    print("正在将向量写入 Pinecone 向量数据库...")
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="page_content"
    )

    vectorstore.add_documents(chunks, batch_size=64) # 分批写入，避免一次性数据量过大太慢
    print("🎉 Embedding + Pinecone ingest 完成！")

if __name__ == "__main__":
    embedding()