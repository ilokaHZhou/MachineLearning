import streamlit as st
from streamlit_chat import message
from core_v2 import run_llm

# 页面标题
st.header("基于知识库文档的RAG (V2)")

# 用户输入框（每次刷新都会重新渲染输入组件）
prompt = st.text_input("Prompt", placeholder="请输入您的问题...")

# 会话状态：保存用户问题历史
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

# 会话状态：保存模型回答历史
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

# 将 source 文档路径美化为多行字符串
def create_sources_string(source_paths):
    if not source_paths:
        return ""
    sources_list = sorted(list(source_paths))
    sources_string = "文档来源:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

# 如果用户输入了 prompt，开始 RAG 处理
if prompt:
    with st.spinner("Generating response..."):  # 显示加载动画
        response = run_llm(prompt)              # 调用 RAG 主逻辑

        # 拿到回答和文档
        answer = response["result"]
        source_docs = response["source_documents"]

        # 提取每个文档的来源文件路径
        sources = {doc.metadata.get("source") for doc in source_docs}

        # 最终展示内容：回答 + 来源列表
        formatted_response = (
            f"{answer}\n\n{create_sources_string(sources)}"
        )

        # 保存历史记录，确保刷新页面后不丢失对话
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)



# 按顺序渲染整个对话历史
if st.session_state["chat_answers_history"]:
    for user_prompt, generated_answer in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"],
    ):
        message(user_prompt, is_user=True)   # 用户消息气泡
        message(generated_answer)            # AI 回复气泡
