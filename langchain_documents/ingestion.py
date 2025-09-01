import asyncio
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from logger import Colors, log_info, log_error, log_success, log_warning, log_header
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

load_dotenv()

# Setup embeddings and vector store
embedder = OllamaEmbeddings(model=os.environ["MODEL"])
vectordatastore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"],
    embedding=embedder,
)


async def fetch_page_content(url: str) -> str:
    """Use Playwright to fetch fully rendered HTML of a JS-heavy page."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            content = await page.content()
            await browser.close()
            log_success(f"Fetched content from {url}")
            return content
    except Exception as e:
        log_error(f"Error fetching {url}: {e}")
        return ""


def extract_documents_from_html(html: str, url: str) -> list:
    """Extract text from HTML and convert to Document objects."""
    if not html.strip():
        log_warning(f"No HTML content to extract for {url}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    if not text:
        log_warning(f"No text extracted from {url}")
        return []

    doc = Document(page_content=text, metadata={"source": url})
    log_info(f"Extracted document from {url}")
    return [doc]


async def main():
    log_header("Starting Document Ingestion Process")

    seed_urls = [
        "https://python.langchain.com/docs/introduction/",
        # Add more URLs if needed
    ]

    all_documents = []

    for url in seed_urls:
        html_content = await fetch_page_content(url)
        documents = extract_documents_from_html(html_content, url)
        all_documents.extend(documents)

    if not all_documents:
        log_warning("No documents extracted. Check seed URLs or extraction logic.")
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
