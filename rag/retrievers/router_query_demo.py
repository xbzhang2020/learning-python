from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
import json

load_dotenv()


class RouterResponse(BaseModel):
    """路由查询的结构化响应"""

    data_source: str = Field(description="选择的数据源名称")
    reason: str = Field(description="选择该数据源的原因")


model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

structured_model = model.with_structured_output(RouterResponse)

routers = ["python_docs", "js_docs", "golang_docs"]

query = input("请输入问题：")
system_msg = SystemMessage(
    f"根据用户问题，选择最适合回答问题的数据源。数据源如下：{routers}"
)
human_msg = HumanMessage(f"{query}")
messages = [system_msg, human_msg]

response = structured_model.invoke(messages)

print("\n=== 结构化输出 ===")
print(
    json.dumps(
        {
            "数据源": response.data_source,
            "原因": response.reason,
        },
        ensure_ascii=False,
        indent=2,
    )
)
