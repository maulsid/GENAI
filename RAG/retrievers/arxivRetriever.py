from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,
    get_ful_documents=True,
)

documents = retriever.invoke("GRU neural networks")

for document in documents:
    print("Metadta:", document.metadata)
    print("Content:", document.page_content[:500])
