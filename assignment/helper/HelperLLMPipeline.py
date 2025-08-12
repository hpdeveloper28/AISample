def llm_pipeline(hf_model, max_length=512, temperature=0.3, top_p=0.95):
    hf_pipeline = None
    # Code starts here

    llm = pipeline(task="text2text-generation", model=hf_model, max_length=max_length)

    hf_pipeline = HuggingFacePipeline(
        pipeline=llm, model_kwargs={"temperature": temperature, "top_p": top_p}
    )

    # Code ends here
    return hf_pipeline
