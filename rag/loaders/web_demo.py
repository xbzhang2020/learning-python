from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_path="https://zh.wikipedia.org/wiki/%E9%BB%91%E7%A5%9E%E8%AF%9D%EF%BC%9A%E6%82%9F%E7%A9%BA",
    bs_kwargs={"parse_only": bs4.SoupStrainer(id="bodyContent")},
)
docs = loader.load()

print(docs)
