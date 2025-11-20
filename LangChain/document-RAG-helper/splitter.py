from loader import load_docs
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

def split_docs_by_markdown_headers(docs):
    # 指定要按哪些标题层级切分
    headers_to_split_on = [
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    new_chunks = []

    for doc in docs:
        markdown_text = doc.page_content

        # 按标题切分成块，每块是 {"content": "...", "metadata": {...}}
        sections = splitter.split_text(markdown_text)

        for section in sections:
            content = section.page_content.strip()

            # 在开头加上 "## " 作为 chunk 标记
            content = f"## {content}"

            # metadata里保留原文件来源
            chunk_doc = Document(
                page_content=content,
                metadata={"source": doc.metadata.get("source")}
            )

            new_chunks.append(chunk_doc)

    return new_chunks


if __name__ == "__main__":
    root = "D:/GitHub_Repos/InterviewAndLeetCode/面试题"
    docs = load_docs(root)
    chunks = split_docs_by_markdown_headers(docs)
    print(f"✅ 按标题切分完成，共 {len(chunks)} 个块")
    print("示例 chunk：\n", chunks[0].page_content[:300])
    print("来源：", chunks[0].metadata)