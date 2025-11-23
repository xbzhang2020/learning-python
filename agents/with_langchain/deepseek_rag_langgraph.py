from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
import os
import operator
from verctor_chroma import vector_store
from langgraph.graph import StateGraph, START, END

load_dotenv()
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    context: str


def retrieve(state: State):
    """Create the index and retrieve the context"""
    query = state["messages"][-1].content
    similar_docs = vector_store.similarity_search(query=query, k=3)
    similar_docs_text = "\n".join([doc.page_content for doc in similar_docs])

    return {"context": similar_docs_text}


def generate(state: State):
    """Generate the response"""
    context = state["context"]
    messages = [
        SystemMessage(
            content=f"你是一名游戏《黑神话：悟空》的专家，请根据以下文档回答用户的问题。如果用户的问题不在文档中，请回答“我不知道”。\n上下文：{context}"
        ),
    ] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# 构建工作流
agent_builder = StateGraph(State)
agent_builder.add_node("retrieve", retrieve)
agent_builder.add_node("generate", generate)
agent_builder.add_edge(START, "retrieve")
agent_builder.add_edge("retrieve", "generate")
agent_builder.add_edge("generate", END)

# 编译工作流
agent = agent_builder.compile()

# 可视化工作流
# image_data = agent.get_graph(xray=True).draw_mermaid_png()
# with open("agent.png", "wb") as f:
#     f.write(image_data)

# 执行工作流
query = input("请输入问题：")
messages = [HumanMessage(content=query)]
result = agent.invoke({"messages": messages})

for m in result["messages"]:
    m.pretty_print()
