from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="data/wukong/人物角色.json",
    jq_schema='.supportCharacters[] | "姓名：" + .name + "，背景：" + .background',
    text_content=True,
)
docs = loader.load()

print(docs)
