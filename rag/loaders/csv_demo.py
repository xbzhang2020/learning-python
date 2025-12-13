from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="data/wukong/黑神话悟空.csv")
docs = loader.load()

print(docs)
