def exact_match(outputs: dict, reference_outputs: dict) -> bool:
    
    actual = outputs["answer"].strip().lower()
    expected = reference_outputs["answer"].strip().lower()

    return actual == expected