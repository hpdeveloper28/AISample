def process_files_in_folder(folder_path, chunk_size, overlap):
    text_chunks = []
    metadata = []
    # Code starts here
    text_splitter = RecursiveCharacterTextSplitter(
        text_splitter.chunk_overlap = overlap
        text_splitter.chunk_size = chunk_size
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        # elif filename.endswith('.docx'):
        #     loader = Docx2txtLoader(filepath)
        #     #loader = TextLoader(file_path=filepath, encoding='utf-8')
        #     #loader = Docx2txtLoader(file_path=filepath)
        else:
            continue

        docs = loader.load()
        print(docs)
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            text_chunks.append(chunk.page_content)
            metadata.append(chunk.metadata)

    # Code ends here
    return text_chunks, metadata