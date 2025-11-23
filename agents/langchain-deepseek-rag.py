# 加载文档
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path="https://zh.wikipedia.org/wiki/%E9%BB%91%E7%A5%9E%E8%AF%9D%EF%BC%9A%E6%82%9F%E7%A9%BA"
)
docs = loader.load()

# 文档切片
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(docs)

# 文档嵌入与向量存储
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vector_store = InMemoryVectorStore(embedding=embeddings_model)
vector_store.add_documents(chunks)

# 向量检索
query = "什么是黑神话：悟空？"
similar_docs = vector_store.similarity_search(query, k=3)
similar_docs_text = "\n".join([doc.page_content for doc in similar_docs])

# 答案生成
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

template = ChatPromptTemplate(
    [
        (
            "system",
            """你是一名游戏《黑神话：悟空》的专家，请根据以下文档回答用户的问题。如果用户的问题不在文档中，请回答“我不知道”。
            上下文：{context}""",
        ),
        ("human", "{query}"),
    ]
)
prompt = template.format(query=query, context=similar_docs_text)

response = model.invoke(prompt)
print(response.content)
