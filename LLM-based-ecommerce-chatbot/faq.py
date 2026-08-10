import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import pandas as pd
import hashlib
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

from config import CSV_PATH, CHROMA_DB_PATH, HASH_FILE, COLLECTION_NAME, TOP_K, GROQ_MODEL

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

embedding_fn = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_last_hash() -> str | None:
    if not os.path.exists(HASH_FILE):
        return None
    with open(HASH_FILE, "r") as f:
        return json.load(f).get("hash")


def save_current_hash(hash_value: str):
    os.makedirs(os.path.dirname(HASH_FILE), exist_ok=True)
    with open(HASH_FILE, "w") as f:
        json.dump({"hash": hash_value}, f)


def build_index():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["question", "answer"])

    questions = df["question"].tolist()
    answers = df["answer"].tolist()
    ids = [f"faq_{i}" for i in range(len(df))]

    print(f"Indexing {len(questions)} FAQ questions...")
    collection.upsert(
        ids=ids,
        documents=questions,
        metadatas=[{"answer": a} for a in answers],
    )
    print(f"Indexed {len(questions)} FAQs into Chroma at '{CHROMA_DB_PATH}'")


def sync_index_if_needed():
    current_hash = get_file_hash(CSV_PATH)
    last_hash = load_last_hash()

    if current_hash == last_hash:
        print("faq.csv unchanged — skipping re-embedding, using existing index.")
        return

    print("faq.csv changed (or first run) — rebuilding index...")
    build_index()
    save_current_hash(current_hash)


def query_faq(user_query: str, top_k: int = TOP_K):
    return collection.query(
        query_texts=[user_query],
        n_results=top_k,
    )


# ---- 7. generate_answer — sends retrieved context + query to Groq, returns final answer ----
def generate_answer(user_query: str, context: list[str]) -> str:
    context_block = "\n".join(f"- {c}" for c in context)

    prompt = f"""You are a helpful ecommerce support assistant.
    Use ONLY the FAQ context below to answer the user's question.
    If the context doesn't contain the answer, say you don't have that information.
 
    Context:
    {context_block}
 
    User question: {user_query}
 
    Answer:"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def faq_chain(user_query: str, top_k: int = TOP_K):
    sync_index_if_needed()
    results = query_faq(user_query, top_k=top_k)
    context = [meta["answer"] for meta in results["metadatas"][0]]
    return generate_answer(user_query, context)


if __name__ == "__main__":
    context = faq_chain("what is your name")
    print(context)
