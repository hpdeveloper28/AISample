def create_vector_store(text_chunks, metadata, embedding_model):
    vectorstore = None
    # Code starts here

    docs = [Document(page_content=text_chunk, metadata=meta) for text_chunk, meta in zip(text_chunks, metadata)]
    vectorstore = FAISS.from_documents(docs, embedding_model)
    # Code ends here
    return vectorstore
