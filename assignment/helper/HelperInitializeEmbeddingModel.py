def initialize_embedding_model(model_name):
    embedder = None
    # Code starts here
    embedder = HuggingFaceEmbeddings(model_name=model_name)

    # Code ends here
    return embedder
