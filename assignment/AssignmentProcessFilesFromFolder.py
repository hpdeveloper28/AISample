import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader


def process_files_in_folder(folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Processes all .pdf and .docx files in a folder, extracts text, and splits it into chunks.

    Args:
        filepath (str): Path to the folder containing files that need to be processed.
        chunk_size (int): The maximum size of each text chunk after splitting.
        chunk_overlap (int): The number of overlapping characters between adjacent chunks.

    Returns:
        tuple: A tuple containing:
            - text_chunks (list): A list of text chunks extracted from the files.
            - metadata (list): A list of dictionaries with metadata for each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = []
    metadata = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        else:
            continue

        docs = loader.load()
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            text_chunks.append(chunk.page_content)
            metadata.append(chunk.metadata)

    return text_chunks, metadata


if __name__ == "__main__":
    process_files_in_folder("/Users/1000060240/PyCharmMiscProject/1-start-here/AISample/assignment")