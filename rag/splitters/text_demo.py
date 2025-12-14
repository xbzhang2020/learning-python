from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("data/shanxi/云冈石窟.txt")
docs = loader.load()

text_splitters = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitters.split_documents(docs)

for chunk in chunks:
    print(chunk.metadata)
    print(chunk.page_content)
    print("-" * 100)
