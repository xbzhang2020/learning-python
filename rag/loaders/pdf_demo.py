from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/wukong/黑神话：悟空-维基百科.pdf")
docs = loader.load()

print(docs[0])