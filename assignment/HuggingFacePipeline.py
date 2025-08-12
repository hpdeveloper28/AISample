from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


def llm_pipeline(hf_model, max_length=512, temperature=0.3, top_p=0.95):
    hf_pipeline = None
    # Code starts here
    llm = pipeline(
        "text2text-generation",
        model=hf_model,
        max_length=max_length,
        # Pass the generation arguments through a dictionary
        generate_kwargs={"temperature": temperature, "top_p": top_p},
    )
    hf_pipeline = HuggingFacePipeline(pipeline=llm)
    # Code ends here
    return hf_pipeline


if __name__ == "__main__":
    # The model name as a string
    model_name = "MBZUAI/LaMini-T5-223M"

    # Call the function with the model name
    llm_pipeline(
        hf_model=model_name,
        max_length=100,  # Shorter generated text
        temperature=0.1,  # Less random output
    )
