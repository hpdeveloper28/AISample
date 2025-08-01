from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def create_vector_store(text_chunks, metadata, embedding_model):
    vectorstore = None
    # Code starts here
    docs = [Document(page_content=chunk, metadata=meta) for chunk, meta in zip(text_chunks, metadata)]
    vectorstore = FAISS.from_documents(docs, embedding_model)
    # Code ends here
    return vectorstore


if __name__ == "__main__":
    # 1. Prepare your arguments
    text_chunks = ["This is the first document.", "Here is another document.", "A third document for our vector store."]
    metadata = [{"source": "file1.txt"}, {"source": "file2.txt"}, {"source": "file3.txt"}]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Call the function with the arguments
    vector_store = create_vector_store(text_chunks, metadata, embedding_model)