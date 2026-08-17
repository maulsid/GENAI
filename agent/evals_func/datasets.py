from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

dataset_name = "simple-qa-eval"

dataset = client.create_dataset(
    dataset_name=dataset_name, description="Simple QA evaluation dataset"
)

examples = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"answer": "Paris"},
    },
    {
        "inputs": {"question": "What is the capital of India?"},
        "outputs": {"answer": "New Delhi"},
    },
    {    
        "inputs": {"question": "What is 2 + 2?"}, 
        "outputs": {"answer": "4"}
    },
]

client.create_examples(dataset_id=dataset.id, examples=examples)
