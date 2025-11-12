import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader

def get_markdown_files(root_dir):
    md_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".md"):
              full_path = os.path.join(dirpath, f)
              md_files.append(full_path)
              # print(full_path)   # 打印文件路径
    return md_files

# 递归获取指定目录下所有 markdown 文件并通过 UnstructuredMarkdownLoader 加载成 Document 列表
def load_docs(root="D:/GitHub_Repos/InterviewAndLeetCode/面试题"):
    md_file_list = get_markdown_files(root)
    print(f"共找到 {len(md_file_list)} 个 markdown 文件。")

    docs = []
    for file in md_file_list:
        loader = UnstructuredMarkdownLoader(file)
        docs.extend(loader.load())
    print(f"文档加载完毕，共加载 {len(docs)} 个文档。")
    # print(docs[0].page_content[:100])   # 打印前100个字符看看
    # print(docs[0].metadata)
    return docs


if __name__ == "__main__":
    root = "D:/GitHub_Repos/InterviewAndLeetCode/面试题"  # 文档目录路径
    load_docs(root)