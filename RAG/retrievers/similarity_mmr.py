from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = [
    Document(page_content="Alice is software engineer and she works in IT"),
    Document(page_content="Alice is hardworking and she is software engineer"),
    Document(page_content="Alice is a person who is confident, reliable and brave"),
    Document(page_content="Alice works in IT"),
    Document(page_content="Alice is Ruckiks sister")
]

embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)


similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("\n===== Similarity Search Results =====\n")

similarity_docs = similarity_retriever.invoke("Who is Alice?")

for doc in similarity_docs:
    print(doc.page_content)


mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

print("\n===== MMR Results =====\n")

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)
    
