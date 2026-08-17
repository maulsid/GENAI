from sentence_transformers import SentenceTransformer

_embeddings = None


def _get_embeddings() -> SentenceTransformer:
    global _embeddings
    if _embeddings is None:
        _embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    return _embeddings


def exact_match(outputs: dict, reference_outputs: dict) -> bool:
    actual = outputs.get("answer", "").strip().lower()
    expected = reference_outputs.get("answer", "").strip().lower()

    return actual == expected


def cosine_similarity(outputs: dict, reference_outputs: dict) -> dict:
    actual = outputs.get("answer", "")
    expected = reference_outputs.get("answer", "")

    actual_vec, expected_vec = _get_embeddings().encode([actual, expected])

    dot = sum(a * b for a, b in zip(actual_vec, expected_vec))
    norm_actual = sum(a * a for a in actual_vec) ** 0.5
    norm_expected = sum(b * b for b in expected_vec) ** 0.5
    score = (
        dot / (norm_actual * norm_expected) if norm_actual and norm_expected else 0.0
    )

    return {"key": "cosine_similarity", "score": float(score)}
