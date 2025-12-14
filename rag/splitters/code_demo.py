from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

TS_CODE = """
function helloWorld(): void {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
"""

ts_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.TS, chunk_size=100, chunk_overlap=20
)
ts_docs = ts_splitter.create_documents([TS_CODE])
for doc in ts_docs:
    print(doc.metadata)
    print(doc.page_content)
    print("-" * 100)
