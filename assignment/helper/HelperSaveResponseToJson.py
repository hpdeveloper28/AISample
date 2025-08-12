def save_response_to_json(question, answer, sources, file_name):
    data = None
    # Code starts here
    data = {"query": question, "answer": answer, "sources": sources}

    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)
    # Code ends here
    return data
