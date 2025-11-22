from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os

load_dotenv()

Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv('DEEPSEEK_API_KEY'))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

documents = SimpleDirectoryReader(input_files=["data/wukong/设定.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("游戏中的故事背景是什么？")
print(response)

index.storage_context.persist()