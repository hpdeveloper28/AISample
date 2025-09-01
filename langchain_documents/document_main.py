from backend.core import run_llm_chain
import streamlit as st

st.header("Document QA with LangChain and Ollama")

prompt = st.text_input("Prompt", placeholder="Enter your question about the documents:")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def create_source_string(sources):
    if not sources:
        return "No sources available."
    source_list = "\n".join(f"- {source}" for source in sources)
    return f"Sources:\n{source_list}"
    pass


if prompt:
    st.spinner("Generating response...")
    response = run_llm_chain(prompt, chat_history=st.session_state["chat_history"] )
    sources = set([doc.metadata["source"] for doc in response["source_documents"]])
    generated_response = f"{response['result']}\n\n {create_source_string(sources)}"
    st.success("Response generated!")
    print(generated_response)

    st.session_state["user_prompt_history"].append(prompt)
    st.session_state["chat_answer_history"].append(generated_response)
    st.session_state["chat_history"].append(("human", prompt))
    st.session_state["chat_history"].append(("ai", generated_response))


if st.session_state["chat_answer_history"]:
    for generated_response, user_prompt in zip(
        st.session_state["chat_answer_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_prompt)
        st.chat_message("assistant").write(generated_response)