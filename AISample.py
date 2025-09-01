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
        "Summarize the following {" + tag_summary + "}.\n\n"
        "Provide the summary and key facts in concise bullet points.\n"
        "{format_instructions}"
    )

    summary_prompt_template = PromptTemplate(
        input_variables=[tag_summary],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    # Connect to the LLaMA 3 model via Ollama
    llm = ChatOllama(model=os.environ["MODEL"])

    chain = summary_prompt_template | llm | summary_parser

    # Run the chain
    res = chain.invoke(input={tag_summary: details})

    # Output the result
    print("========================================================================")
    print(res)
    for fact in res.facts:
        print(f"- {fact}")
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
        """"1. Daydreams.
I was sitting in class when I first saw it.

	Miss Weaver had been my teacher for a few months, and was known around Stagwood Elementary for the stack of black hair that rose a foot above her head. Before the school year started, I’d heard a few rumors about her, and within a week I realized that they were all true. For one thing, she did, in fact, wear the same outfit every day; the colors changed, but she always had on striped pants and a striped jacket. For another thing, she was mind-numbingly boring. It was the kind of boring that made your eyes shut without permission. The biggest problem, though, was the stories. She was obsessed with tales of former students who had become some kind of famous. The first couple of times weren’t bad, maybe even kind of interesting. But, by the second week of school she had started repeating herself, just like her outfits. 

	By then, I knew all the stories by heart. The professional football player who got excellent marks in Math. The State Senator who was a teacher’s pet. I knew every word. So, instead of trying my hardest to listen, I spent most of class drawing in my notebook. 

	Most days, I drew imaginary places and then spent the rest of the time whipped up creatures to live there. They’d have horns where horns don’t go, fur where scales should be, and all the wings. They were odd. And they each had a story that I wanted to tell. But, on that day, I never even got to the first pair of wings. I had barely gotten started when it appeared, and changed Stagwood, and me, forever.

	If it had chosen to press its little green face against any other window, I might not have seen it. And if I hadn’t been trying to decide whether my dragon should have four legs or two, I might not have looked out window at that exact moment, dropping my pencil on the page. 

	It was a frog. And it was staring right at me.

	I couldn’t stop looking at the frog and it couldn’t stop looking back. We were locked in a staring contest. Maybe some frogs blinked, but with its eyes smushed against the glass, this one didn’t. Stagwood Forest was just beyond the schoolyard and it was riddled with frogs, but they always avoided people. I knew right away, in a way that I could think better than I could say, that this frog was different.

	I tried to listen back in to Miss Weaver, just in time to hear the end of her story about Martin Shandals, the now-famous comedian. Martin had transferred schools half way through the year, so I always felt to me like that one shouldn’t count. We were supposed to be learning long division, but something had reminded her of Martin. I knew exactly what bad joke she would end the story with, but much less about long division. 

	“Whenever he acted up in class I’d say, ‘we’ve got a real comedian on our hands don’t we?’ And I was right!” she said with a guffaw. 

	I was certain that Miss Weaver would see the frog within moments, but I was wrong. Nobody did. More importantly, when I looked again to see if it was still there, I noticed something new. Something shiny. And when I realized what it was, I forgot all about class, and Miss Weaver, and Martin Shandals. There was no denying it: the frog had put on a tiny pair of glasses. 

	I wanted to lecture it, to explain that frogs don’t wear glasses. It bothered me that it didn’t already know that. On top of that, it had been staring at me for at least five minutes by then. And that seemed to be bordering on rude. Could a frog even be rude? I wasn’t sure. But, the bigger question was why it was so interested in me.

	I wasn't the type of kid who got attention. Teachers always wrote “needs to participate more” on my report cards (with a smiley face to make my parents feel better). I never got into trouble and barely ever stood out on purpose. A few years earlier, I accidentally peed my pants because my zipper had gotten stuck in the bathroom at the last moment. I tried to convince everyone that I had fallen into a puddle at recess. The custodian, Mr. Salazar, charged outside with a mop and brought me with him to point out the puddle. My guess is that we wasted a half-hour looking around at the dry gravel. Luckily, my mom dropped off some new clothes and nobody really noticed my wardrobe change (…or that it hadn’t rained in weeks). 

	That’s how it was. Whether I did something spectacular or sneezed myself out of a chair, nobody cared, and almost nobody said my name. As far as school was concerned, all those things had happened to “some kid”. So, why would a frog with glasses jump up on a windowsill to stare at "some kid"? 

	Teachers, on the other hand, were a different story. Once her lesson started back up, it didn’t take Miss Weaver long to realize that I wasn’t paying attention. She called me up to the blackboard to make an example out of me. 

	“Since you don’t feel the need to listen, why don’t you solve a problem on the board instead?” she said, sitting down at her desk. 

	My stomach did a flip. Then it did a flop. The problem would take a minute or two to solve, and being in front of the class always made me nervous. How could I be expected to do anything when there was a spectacled frog staring me down?

	I walked to the right of the equation on the board so that I could check on the frog easily in secret. Despite the distraction, I did my best to focus. But, it wasn’t easy. Halfway through, I saw the frog move towards the front of the classroom. It stopped at the window next to Miss Weaver’s desk. It took me a moment to figure out what it was doing, and another to believe it. It was trying to lift the window!

	Finishing the problem became almost impossible. I made a mistake and then quickly erased it. By the time I looked over again, the window was open. Why should that surprise me? Of course a frog with glasses would also be super strong. The window was only open an inch, but that was enough for it to slip through. I dropped the chalk, and some of my classmates laughed. Bending down to pick it up, I tried convincing myself that when I stood back up again the frog would be gone. “It’s not there. I just think it’s there.”

	When I straightened up, the frog was sitting on Miss Weaver’s left shoulder. 

	This was a brave frog.

	Her head blocked the class from seeing it, and I realized that I was still the only one who could. Either the frog was real or my imagination had outdone itself. It wasn’t all that surprising that Miss Weaver didn’t feel it there, because the shoulder pads inside her orange striped jacket were large and fluffy. I had heard that she rested her head on them like pillows during her breaks. I tried to remind myself of the situation. There was a frog sitting on Miss Weaver’s shoulder and nobody else knew it. And I was supposed to be doing math.

	Now that it was closer, I could see the frog better. It didn’t look like some new species of frog to me. It looked like every other frog I had seen (except for the glasses). I wondered if they made contact lenses small enough for a frog. But, it wasn’t the right time to worry about frog vision- that would be a job for a frog eye doctor, anyway.

	I had daydreams all the time when I was drawing, and sometimes I got lost in them. It really was possible, I thought, that my imagination had just carried me off. I tried one last time to explain the frog away as part of an impressive daydream. I concentrated hard, finished the problem, and put the chalk down. The frog couldn’t be real. I shook my head confidently. 

	When I turned to Miss Weaver, I saw the frog look me square in the eyes and nod. A moment later, it disappeared into Miss Weaver’s hair."""
    )
