from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


# =========================================================
# STEP 1: Load environment variables
# =========================================================
# Loads values from the .env file, such as MISTRAL_API_KEY.

load_dotenv()


# =========================================================
# STEP 2: Get the user's question
# =========================================================

query = input("Ask me anything: ")


# =========================================================
# STEP 3: Initialize the embedding model
# =========================================================
# The embedding model converts the user's question into a
# numerical vector.
#
# Important:
# Use the same embedding model that was used when creating
# and storing the document embeddings.

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu",
    },
    encode_kwargs={
        "normalize_embeddings": False,
    },
)


# =========================================================
# STEP 4: Connect to the existing Chroma vector database
# =========================================================
# This does not load and split the PDF again.
# It opens the existing persisted Chroma database.

db_path = Path(__file__).resolve().parent / "chroma-db"

vectorstore = Chroma(
    collection_name="chromavectorstore",
    persist_directory=str(db_path),
    embedding_function=embedding_model,
)


# =========================================================
# STEP 5: Create the retriever
# =========================================================
# The retriever searches the vector database for chunks
# related to the user's question.
#
# k = final number of chunks returned
# fetch_k = initial number of candidate chunks
# lambda_mult = balance between similarity and diversity

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)


# =========================================================
# STEP 6: Retrieve relevant document chunks
# =========================================================
# The query is converted into an embedding.
# Chroma compares it with the stored document embeddings
# and returns the most relevant chunks.

results = retriever.invoke(query)


# =========================================================
# STEP 7: Display the retrieved chunks
# =========================================================
# This step is mainly useful for debugging and understanding
# what information is being given to the LLM.

print("\n------------------------------------")
print("Retrieved document chunks")
print("------------------------------------")

for index, document in enumerate(results, start=1):
    print(f"\nResult {index}")
    print(document.page_content)
    print("Metadata:", document.metadata)


# =========================================================
# STEP 8: Combine the retrieved chunks into one context
# =========================================================
# The LLM receives this combined context.

context = "\n\n".join(
    document.page_content
    for document in results
)

print("\n------------------------------------")
print("Combined context")
print("------------------------------------")
print(context)


# =========================================================
# STEP 9: Initialize the LLM
# =========================================================
# The LLM generates the final natural-language answer using
# the retrieved document context.

llm = ChatMistralAI(
    model="mistral-small-2603",
    temperature=0,
    max_retries=2,
)


# =========================================================
# STEP 10: Create the prompt template
# =========================================================
# The prompt tells the LLM:
# 1. What role it has
# 2. What context it must use
# 3. What question it must answer
# 4. What to say when the answer is unavailable

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use only the provided context to answer the question.
Do not use outside knowledge.

If the answer is not present in the context, say:
"I don't know. I am sorry, Mauli."

You may answer in a funny way, but the answer must remain accurate.
""",
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
""",
        ),
    ]
)


# =========================================================
# STEP 11: Insert the context and question into the prompt
# =========================================================
# This creates the final formatted messages that will be
# sent to the Mistral model.

formatted_prompt = prompt.format_prompt(
    context=context,
    question=query,
)


# =========================================================
# STEP 12: Send the prompt to the LLM
# =========================================================

response = llm.invoke(formatted_prompt)


# =========================================================
# STEP 13: Display the final answer
# =========================================================

print("\n------------------------------------")
print("Final answer")
print("------------------------------------")
print(response.content)