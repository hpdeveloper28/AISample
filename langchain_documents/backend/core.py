from typing import List, Dict, Any

from dotenv import load_dotenv
import os

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_pinecone import PineconeVectorStore

load_dotenv()
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings


def run_llm_chain(query: str, chat_history=List[Dict[str, Any]]):
    llm = ChatOllama(model=os.environ["MODEL"])
    embedder = OllamaEmbeddings(model=os.environ["MODEL"])

    docsearch = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embedder,
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_document_chain = create_stuff_documents_chain(
        llm, prompt=retrieval_qa_chat_prompt
    )

    rephase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm = llm, retriever=docsearch.as_retriever(), prompt=rephase_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_document_chain
    )

    response = retrieval_chain.invoke(input={"input": query, "chat_history": chat_history})
    new_response = {
        "result": response["answer"],
        "source_documents": response["context"],
        "query": response["input"],
    }
    return new_response


if __name__ == "__main__":
    query = "What is LangChain?"
    response = run_llm_chain(query)
    print(response["result"])
