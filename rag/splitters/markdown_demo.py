from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

with open("data/wukong/黑悟空版本介绍.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
)
markdown_chunks = markdown_splitter.split_text(markdown_text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
)
chunks = text_splitter.split_documents(markdown_chunks)

for chunk in chunks:
    print(chunk.metadata)
    print(chunk.page_content)
    print("-" * 100)
