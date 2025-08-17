from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

if __name__ == "__main__":
    load_dotenv()

    # Load text data from a file
    loader = TextLoader("medium_blog.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks of text from the document.")

    embeddings = OllamaEmbeddings(model=os.environ["MODEL"])

    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )

    print("Finish")
