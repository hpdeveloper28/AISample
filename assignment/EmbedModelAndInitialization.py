from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def initialize_embedding_model(model_name: str):
    embedder = None
    # Code starts here
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    # Code ends here
    return embedder


if __name__ == "__main__":
    initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")