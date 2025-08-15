from typing import Tuple
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import linkedin_lookup_agent
from tools.output_parser import summary_parser, Summary

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
    linkedin_data = scrape_linkedin_profile(
        profile_url="https://gist.githubusercontent.com/hpdeveloper28/5b66ccb34c2dc02ebea197bfe3d3680e/raw/52020e549c901224d372396b35318957fbb3bd36/sample_linkedin.json"
    )

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


def get_crux_of_lengthy_content(details: str):

    tag_summary = "summary"
    # Create the summary prompt template
    summary_template = "Summarize the following {" + tag_summary + "}"

    summary_prompt_template = PromptTemplate(
        input_variables=[tag_summary], template=summary_template
    )

    # Connect to the LLaMA 3 model via Ollama
    llm = ChatOllama(
        model=os.environ["MODEL"]
    )  # e.g., "llama3" or "llama3:8b-instruct"

    # Build the chain
    chain = summary_prompt_template | llm | StrOutputParser()

    # Run the chain
    res = chain.invoke(input={tag_summary: details})
    print("========================================================================")
    # Output the result
    print(res)


def get_crux_of_lengthy_content_with_output_parser(details: str) -> str:

    tag_summary = "summary"
    # Create the summary prompt template
    summary_template = (
        "Summarize the following {" + tag_summary + "}" "\n{format_instructions}"
    )

    summary_prompt_template = PromptTemplate(
        input_variables=[tag_summary],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    # Connect to the LLaMA 3 model via Ollama
    llm = ChatOllama(
        model=os.environ["MODEL"]
    )  # e.g., "llama3" or "llama3:8b-instruct"

    # Build the chain
    # chain = summary_prompt_template | llm | StrOutputParser()

    chain = summary_prompt_template | llm | summary_parser

    # Run the chain
    res = chain.invoke(input={tag_summary: details})

    # Output the result
    print("========================================================================")
    print(res)
    return res


if __name__ == "__main__":
    # This function loads the data from .env file

    # This is use for live model
    # llm = ChatOpenAI(temperature=0, model_name="o3-pro")

    # find_relevant_information()
    # find_linkedin_profile()
    # get_linkedin_username("Hiren Patel Hexaware")

    # get_crux_of_lengthy_content(
    #     "This night is cold in the kingdom I can feel you fade away from the kitchen to the bathroom sink in Your steps keep me awake Don't cut me down throw me out leave me in a waste I once was in man with dignity and grace Now I'm slipping through the cracks of your cold embrace so please please Could you find a way to let me down slowly? A little sympathy I hope you can show me If you want to go, then I'll be so lonely If you leave him, baby let me down slowly If you want to go, then I'll be so lonely If you leave him, baby let me down slowly Cool skin drag my feet on the tile As I'm walking down the corridor We have been talked in a while So I'm looking for an open door Don't cut me down through me I've been in a waste I once was a man with skinny and grace Now I'm slipping through the cracks Should be cold and raised so please Please Could you find a way to let me down slowly? A little sympathy I hope you can show me If you wanna go then I'll be so lonely If you leave me baby let me down slowly Let me down down Let me down down Let me down down Let me down If you wanna go then I'll be so lonely If you leave me baby let me down slowly And I can stop myself from falling Don't come And I can stop myself from falling Don't come And I can stop myself from falling Don't come And I can stop myself from falling Don't come Could you find a way to let me down slowly? A little sympathy I hope you can show me If you wanna go then I'll be so lonely If you leave me baby let me down slowly Let me down down If you wanna go then I'll be slowly If you leave me baby let me down slowly And if you wanna go then I'll be slowly If you leave me baby let me down slowly And if you wanna go then I'll be slowly"
    # )

    get_crux_of_lengthy_content_with_output_parser(
        "This night is cold in the kingdom I can feel you fade away from the kitchen to the bathroom sink in Your steps keep me awake Don't cut me down throw me out leave me in a waste I once was in man with dignity and grace Now I'm slipping through the cracks of your cold embrace so please please Could you find a way to let me down slowly? A little sympathy I hope you can show me If you want to go, then I'll be so lonely If you leave him, baby let me down slowly If you want to go, then I'll be so lonely If you leave him, baby let me down slowly Cool skin drag my feet on the tile As I'm walking down the corridor We have been talked in a while So I'm looking for an open door Don't cut me down through me I've been in a waste I once was a man with skinny and grace Now I'm slipping through the cracks Should be cold and raised so please Please Could you find a way to let me down slowly? A little sympathy I hope you can show me If you wanna go then I'll be so lonely If you leave me baby let me down slowly Let me down down Let me down down Let me down down Let me down If you wanna go then I'll be so lonely If you leave me baby let me down slowly And I can stop myself from falling Don't come And I can stop myself from falling Don't come And I can stop myself from falling Don't come And I can stop myself from falling Don't come Could you find a way to let me down slowly? A little sympathy I hope you can show me If you wanna go then I'll be so lonely If you leave me baby let me down slowly Let me down down If you wanna go then I'll be slowly If you leave me baby let me down slowly And if you wanna go then I'll be slowly If you leave me baby let me down slowly And if you wanna go then I'll be slowly"
    )
