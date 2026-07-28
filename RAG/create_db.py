## load pdf
# split into chunks
# create embeddings 
# store in databse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
## 1> load pdf
loader = PyPDFLoader("document_loaders/GRU.pdf")
documents = loader.load()
#print(documents[0].page_content)

#2> split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)
chunks = splitter.split_documents(documents)

#3> create db and store embeddings 
Chroma.from_documents(documents= chunks, collection_name="chromavectorstore", persist_directory="chroma-db", embedding=hf)
#print(query_embeddings)