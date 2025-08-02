from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def llm_pipeline(hf_model, max_length=512, temperature=0.3, top_p=0.95):
    hf_pipeline = None
    # Code starts here
    llm = pipeline(
        "text2text-generation",
        model=hf_model,
        max_length=max_length
    )

    hf_pipeline = HuggingFacePipeline(
        pipeline=llm,
        model_kwargs={'temperature': temperature, 'top_p': top_p}
    )
    # Code ends here
    return hf_pipeline

def conversational_chain(llm, vector_store, question, memory_key, k):
    answer = None
    sources = None
    # Code starts here
    memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True, output_key="answer")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        output_key="answer",
        return_source_documents=True
    )

    result = qa_chain.invoke({"question": question})

    answer = result['answer']
    sources = [doc.page_content for doc in result['source_documents']]
    # Code ends here
    return answer, sources


if __name__ == "__main__":
    model_name = "MBZUAI/LaMini-T5-223M"

    # Call the function with the model name
    llm_model = llm_pipeline(
        hf_model=model_name,
        max_length=100,  # Shorter generated text
        temperature=0.1  # Less random output
    )
    text_chunks = ["..."]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    user_question = "What is the capital of France?"
    history_key = "chat_history"
    num_documents = 3

    conversational_chain(
        llm=llm_model,
        vector_store=vector_store,
        question=user_question,
        memory_key=history_key,
        k=num_documents
    )