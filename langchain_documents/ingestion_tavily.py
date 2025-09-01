import asyncio
import ssl
import os
from dotenv import load_dotenv
from typing import Any, Dict, List
import certifi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import Colors, log_info, log_error, log_success, log_warning, log_header
import asyncio
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

load_dotenv()

# SSL setup
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Embeddings and vector store
embedder = OllamaEmbeddings(model=os.environ["MODEL"])
vectordatastore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"],
    embedding=embedder,
)

# Tavily tools
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_pages=1000, max_breadth=20)
tavily_crawl = TavilyCrawl()


async def manual_crawl_and_extract(url: str) -> list[Document]:
    """Manually crawls a URL using requests and BeautifulSoup."""
    try:
        log_info(f"Manually crawling URL: {url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # This will raise an HTTPError if the response was an error

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all readable text from the body of the HTML
        page_content = soup.get_text(separator=" ", strip=True)

        if not page_content:
            log_warning(f"No text content found in manual crawl for {url}")
            return []

        document = Document(page_content=page_content, metadata={"source": url})

        log_success(f"Successfully extracted content from {url}")
        return [document]

    except requests.exceptions.RequestException as e:
        log_error(f"Error during manual crawl for {url}: {e}")
        return []


async def main():
    log_header("Starting Document Ingestion Process")

    seed_urls = [
        "https://python.langchain.com/docs/introduction/",
        # Add more URLs here if needed
    ]
    all_documents = []

    for url in seed_urls:
        docs = await manual_crawl_and_extract(url)
        all_documents.extend(docs)

    if not all_documents:
        log_warning(
            "No documents extracted. Check your seed URLs or extraction functions."
        )
        log_header("Document Ingestion Process Completed")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(all_documents)
    log_info(f"Split into {len(split_documents)} chunks.")

    # Add to vector store
    vectordatastore.add_documents(split_documents)
    log_success("Documents ingested into vector store successfully.")

    log_header("Document Ingestion Process Completed")


if __name__ == "__main__":
    asyncio.run(main())
