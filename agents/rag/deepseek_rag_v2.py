"""
使用 LangChain 和 Chroma 实现 RAG 流程
1. 向量检索
2. 答案生成
"""

query = input("请输入问题：")

# 向量检索
from verctor_chroma import chroma_wukong_db

similar_docs = chroma_wukong_db.similarity_search(query, k=3)
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

for chunk in model.stream(prompt):
    print(chunk.text, end="")
print("")
