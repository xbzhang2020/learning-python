from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/wukong/设定.txt")
docs = loader.load()

for doc in docs:
    print(doc.metadata)
    print(doc.page_content)
    print("-" * 100)