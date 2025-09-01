import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain import hub

load_dotenv()

if __name__ == "__main__":
    pdf_path = "sample_blood_report.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    embedder = OllamaEmbeddings(model=os.environ["MODEL"])
    vectorstore = FAISS.from_documents(docs, embedder)
    vectorstore.save_local("faiss_index_react")

    new_vector_data_store = FAISS.load_local(
        "faiss_index_react", embedder, allow_dangerous_deserialization=True
    )
    # query = "What is the hemoglobin level mentioned in the report?"
    # results = new_vector_data_store.similarity_search(query)
    # for i, doc in enumerate(results):
    #     print(f"Result {i+1}:")
    #     print(doc.page_content)
    #     print("-" * 50)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    llm = ChatOllama(model=os.environ["MODEL"])

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrieval_qa_chain = create_retrieval_chain(
        new_vector_data_store.as_retriever(), combine_docs_chain
    )
    query = "What is the Pulmonary details mentioned in the report?"
    result = retrieval_qa_chain.invoke(input={"input": query})
    print(result["answer"])
