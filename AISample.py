from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import linkedin_lookup_agent

tag = "information"
load_dotenv()

def find_relevant_information():
    subject = """Rahul Dravid"""
    print("Finding relevant information for " + subject)

    # Normal data find
    summary_template = "Provide the {" + tag + "}"

    summary_prompt_template = PromptTemplate(
        input_variables=[tag], template=summary_template
    )

    llm = ChatOllama(model=os.environ["MODEL"])

    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={tag: subject})

    print(res)

    print("Found relevant information for " + subject)


def find_linkedin_profile():
    # LinkedIn profile find
    summary_template = "Provide the LinkedIn profile {" + tag + "}"

    summary_prompt_template = PromptTemplate(
        input_variables=[tag], template=summary_template
    )

    # This is for local model (Used Ollama)
    llm = ChatOllama(model=os.environ["MODEL"])

    chain = summary_prompt_template | llm | StrOutputParser()

    linkedin_data = scrape_linkedin_profile(
        "https://gist.githubusercontent.com/hpdeveloper28/5b66ccb34c2dc02ebea197bfe3d3680e/raw/52020e549c901224d372396b35318957fbb3bd36/sample_linkedin.json"
    )

    res = chain.invoke(input={tag: linkedin_data})

    print(res)


def get_linkedin_username(name: str) -> str:
    print("Fetching linkedin username")
    linkedin_username = linkedin_lookup_agent(name=name)
    print(linkedin_username)
    linkedin_data = scrape_linkedin_profile(profile_url="https://gist.githubusercontent.com/hpdeveloper28/5b66ccb34c2dc02ebea197bfe3d3680e/raw/52020e549c901224d372396b35318957fbb3bd36/sample_linkedin.json")

    # Normal data find
    summary_template = f"""
    You are given the URL which is having JSON data representing a person's online profile.
    Please provide only {tag} Education and country details. I don't want any personal information
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(model=os.environ["MODEL"], temperature=0)
    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": linkedin_data})

    print(res)


if __name__ == "__main__":
    # This function loads the data from .env file

    # This is use for live model
    # llm = ChatOpenAI(temperature=0, model_name="o3-pro")

    # find_relevant_information()
    # find_linkedin_profile()
    get_linkedin_username("Hiren Patel Hexaware")
