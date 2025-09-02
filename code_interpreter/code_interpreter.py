from dotenv import load_dotenv
from langchain import hub

load_dotenv()
import os
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool


def main():
    print("Hello, Code Interpreter!")

    instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
"""
    base_prompt = hub.pull("langchain-ai/react-agent-template")

    tools = [PythonREPLTool()]
    prompt = base_prompt.partial(
        instructions=instructions,
        tools="\n".join([t.name for t in tools]),
        tool_names=", ".join([t.name for t in tools])
    )

    llm = ChatOllama(model=os.environ["MODEL"], temperature=0)

    agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke(
        input={
            "input": "Generate and save in current directory 15 QR codes that points to www.udemy.com/course/langchain, you have QR code package installed already."
        }
    )


if __name__ == "__main__":
    main()
