from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("data/shanxi/云冈石窟.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", " ", "", "。", "，"]
)
chunks = text_splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.metadata)
    print(chunk.page_content)
    print("-" * 100)
