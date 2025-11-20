import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

from sentence_transformers import CrossEncoder

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "test"
# reranker = CrossEncoder("BAAI/bge-reranker-large")
reranker = CrossEncoder("./models/bge-reranker-large")

def rerank_docs(query, docs):
    """
    使用 CrossEncoder 对检索到的文档进行重排序
    - Query 和 Doc 放在同一个模型里，让模型交叉注意（cross-attention）
    - 模型不仅看相似度，还看语义逻辑、句法结构、推理关系
    - 大幅改善“表面相似但不相关”的问题
    """
    pairs = [[query, d.page_content] for d in docs] # 把原始query和文档内容组成对，
    scores = reranker.predict(pairs)  # 得分越高越相关

    # 将 doc 和 score 绑定排序
    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 返回前5个更精准的文档
    return [doc for doc, score in ranked[:5]]


def run_llm(query: str):
    # 1) Embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

    # 2) 初始化 Pinecone 向量检索器
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="page_content"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # 检索文档
    docs = retriever.invoke(query)

    # rerank重排文档
    docs = rerank_docs(query, docs)

    for i, doc in enumerate(docs):
        print(f"\n=== 检索到文档 #{i+1} ===")
        print("✅ 来源：", doc.metadata.get("source"))
        print("✅ 内容：", doc.page_content[:200], "...")

    # 3) DeepSeek 作为Chat回答的生成模型
    chat = ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
        verbose=True,
    )

    # 4) RAG Prompt
    template = """
        你是一个专业问答助手。请直接回答用户问题，不要描述你自己，也不要解释你的推理过程。

        以下是与你问题最相关的资料（可能包含多个文档片段）：

        <context>
        {context}
        </context>

        要求：
        - 直接回答问题，不要以“根据上下文”“根据提供的信息”开头
        - 若资料中没有答案，回复：“相关信息不在知识库中”

        用户问题：{question}
    """
    prompt = PromptTemplate.from_template(template)

    # 5) 使用 LCEL 构建 RAG Chain（LangChain 1.0.5）
    rag_chain = (
        RunnableParallel({
            "context": RunnableLambda(lambda _: docs),
            "question": RunnablePassthrough(),
        })
        | prompt
        | chat
        | StrOutputParser()     # 输出为纯文本
    )

    # 6) 调用 Deepseek 模型拿到结果
    rag_answer = rag_chain.invoke(query)

    print(f"result: {rag_answer}")

    new_result = {
        "query": query,
        "result": rag_answer,
        "source_documents": docs
    }
    return new_result


if __name__ == "__main__":
    res = run_llm("简单说说Vue的生命周期？")
    print(res)
