import os
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
jina_api_key = os.getenv("JINA_API_KEY")


INDEX_NAME = "dochelper"


# def jina_rerank(query: str, documents: list[str], top_n=3):
#     url = "https://api.jina.ai/v1/rerank"

#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {jina_api_key}",
#     }

#     payload = {
#         "model": "jina-reranker-v2-base-multilingual",   # 中文最强版本
#         "query": query,
#         "documents": documents,
#         "top_n": top_n,
#     }

#     response = requests.post(url, json=payload, headers=headers)
#     data = response.json()

#     # 返回排序后的文档列表（按 index 排序）
#     results = data["results"]
#     return results


def run_llm(query: str):
    # 1) Embedding（来自 Ollama）
    embeddings = OllamaEmbeddings(model="EntropyYue/jina-embeddings-v2-base-zh")

    # 2) 初始化 Pinecone 向量检索器
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="page_content"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    docs = retriever.invoke(query)

    for i, doc in enumerate(docs):
        print(f"\n=== 检索到文档 #{i+1} ===")
        print("✅ 内容：", doc.page_content[:200], "...")
        print("✅ 来源：", doc.metadata.get("source"))

    # 3) DeepSeek Chat 作为生成模型
    chat = ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
        verbose=True,
    )

    # 4) RAG Prompt
    template = """
      请根据以下提供的上下文内容回答用户问题。

      <context>
      {context}
      </context>

      如果答案不在上下文中，请回答：“相关信息不在知识库中”， 并给出上下文内容。

      用户问题：{question}
    """
    prompt = PromptTemplate.from_template(template)

    # 5) 使用 LCEL 构建 RAG Chain（LangChain 1.0.5）
    rag_chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough(),
        })
        | prompt
        | chat
        | StrOutputParser()     # 输出为纯文本
    )

    # 6) 调用模型
    result = rag_chain.invoke(query)
    return result


if __name__ == "__main__":
    # res = run_llm("简单说说React的setState批量更新过程？")
    res = run_llm("说说JSX是啥？")
    print(res)
