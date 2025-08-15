from typing import Union, List

from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_ollama import ChatOllama
import os
from langchain.agents import tool
from langchain.tools  import Tool

load_dotenv()


@tool()
def get_text_length(text: str) -> int:
    """
    Get the length of the text.
    """
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name '{tool_name}' not found.")



if __name__ == "__main__":
    # Example usage
    sample_text = "This is a \n sample text."
    length = get_text_length.invoke(input={"text": sample_text})
    print(f"The length of the text is: {length}")

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:"""

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join(tool.name for tool in tools),
    )

    llm = ChatOllama(temperature=0, model=os.environ["MODEL"], stop=["\nObservation:"])

    agent = {"input": lambda x:x["input"]} | prompt | llm | ReActSingleInputOutputParser()

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of the text 'This is a sample text.'?"})

    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"Observation: {observation}")
