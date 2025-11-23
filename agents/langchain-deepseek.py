from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv('DEEPSEEK_API_KEY'),
)

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")
messages = [system_msg, human_msg]

for chunk in model.stream(messages):
    print(chunk.text, end="")