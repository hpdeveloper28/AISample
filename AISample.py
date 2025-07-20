from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

subject = """Rahul Dravid"""
tag = "information"

if __name__ == "__main__":
    # This function loads the data from .env file
    load_dotenv()
    print("Finding relevant information for "+ subject)

    summary_template = "Provide the {" + tag + "}"

    summary_prompt_template = PromptTemplate(
        input_variables=[tag], template=summary_template
    )
    # This is use for live model
    # llm = ChatOpenAI(temperature=0, model_name="o3-pro")

    # This is for local model (Used Ollama)
    llm = ChatOllama(model=os.environ["MODEL"])

    chain = summary_prompt_template | llm

    res = chain.invoke(input={tag: subject})

    print(res.pretty_print())

    print("Found relevant information for " +subject)
