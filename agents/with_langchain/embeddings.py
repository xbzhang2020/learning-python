from langchain_huggingface.embeddings import HuggingFaceEmbeddings

huggingface_embeddings_zh = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
