from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embeddings import huggingface_embeddings_zh

vector_store = Chroma(
    embedding_function=huggingface_embeddings_zh,
    persist_directory="chroma_wukong_db",
)

def add_documents_to_chroma():
    # 加载文档
    loader = WebBaseLoader(
    web_path="https://zh.wikipedia.org/wiki/%E9%BB%91%E7%A5%9E%E8%AF%9D%EF%BC%9A%E6%82%9F%E7%A9%BA"
    )
    docs = loader.load()

    # 文档切片
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)

    # 文档嵌入与向量存储
    vector_store.add_documents(chunks)

if __name__ == "__main__":
    add_documents_to_chroma()
    print("Documents added to Chroma")