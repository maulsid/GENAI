from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

query = input("Ask me anything: ")

db_path = Path(__file__).resolve().parent / "chroma-db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

vectorstore = Chroma(
    collection_name="chromavectorstore",
    persist_directory=str(db_path),
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)

results = retriever.invoke(query)

print("------------------------------------")

for index, document in enumerate(results, start=1):
    print(f"\nResult {index}")
    print(document.page_content)
    print("Metadata:", document.metadata)

context = "\n\n".join(document.page_content for document in results)

print("\nCombined context:")
print(context)

llm = ChatMistralAI( 
 model="mistral-small-2603", 
 temperature=0, 
 max_retries=2, 
 ) 

#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
        """You are a helpful AI assistant.
        Use ONLY the provided context to answer the question.
        If the answer is not present in the context,
        Answer in funny way and say "i dont know, i am sorry mauli"
       """
        ),
        (
            "human",
        """Context:{context}
           Question:{question}
        """
        )
    ]
)


formatted_prompt = prompt.format_prompt(
    context=context,
    question=query,
)

response = llm.invoke(formatted_prompt)
print("------------------------------------")

print(response.content)
