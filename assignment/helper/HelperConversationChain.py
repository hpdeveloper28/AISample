def conversational_chain(llm, vector_store, question, memory_key="chat_history", k=3):
    answer = None
    sources = []
    # Code starts here
    memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True, output_key="answer")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     retriever=retriever,
                                                     memory=memory,
                                                     return_source_documents=True,
                                                     output_key="answer")

    result = qa_chain.invoke({"question": question})

    answer = result['answer']
    print(f"Result: {result['source_documents']}")
    # sources = [doc.metadata.get('source') for doc in result.get('source_documents')]

    # # for doc in result["source_documents"]:
    # #     if hasattr(doc, "metadata"):
    # #         sources.append(doc.metadata.get("source", "Unknown"))
    # #     else:
    # #         print("Issue in parsing")
    # Code ends here
    return answer, sources