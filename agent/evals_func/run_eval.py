from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import Client
import os

from evals_func.utils import cosine_similarity, exact_match

load_dotenv()

client = Client()
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=os.environ.get("GROQ_API_KEY"),
    reasoning_format="hidden",
)


def target(inputs: dict) -> dict:
    response = llm.invoke(inputs["question"])
    print(f"Q: {inputs['question']}\nA: {response.content}\n")
    return {"answer": response.content}


if __name__ == "__main__":
    client.evaluate(
        target,
        data="simple-qa-eval",
        evaluators=[exact_match, cosine_similarity],
        experiment_prefix="simple-qa",
    )
