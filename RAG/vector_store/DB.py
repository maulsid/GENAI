from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader

load_dotenv()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
    )
#query_embeddings = hf.embed_query(query)
loader = PyPDFLoader("document_loaders/GRU.pdf")
documents = loader.load()

Chroma.from_documents(documents= documents, collection_name="chromavectorstore", persist_directory="chroma-db", embedding=hf)
#print(query_embeddings)