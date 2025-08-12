import json


def save_response_to_json(
    question: str, answer: str, sources: list, file_name: str = "output.json"
):
    output = None
    # Code starts here
    output = {"query": question, "answer": answer, "sources": sources}
    with open(file_name, "w") as f:
        json.dump(output, f, indent=4)
    # Code ends here
    return output


if __name__ == "__main__":
    question_text = "What is the capital of France?"
    generated_answer = "The capital of France is Paris."
    source_list = [
        "Document A: France is a country in Western Europe. Its capital is Paris.",
        "Document B: Paris is known for the Eiffel Tower.",
    ]

    # Call the function with all the required arguments
    save_response_to_json(
        question=question_text,
        answer=generated_answer,
        sources=source_list,
        file_name="conversation_output.json",
    )
