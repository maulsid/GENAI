from langchain_community.document_loaders import TextLoader, PyPDFLoader

# development of RAG plan(How RAG works)
#1. user uploads material including pdfs, links notes etc
#2. The system loads documents using document loaders.Goal:Convert raw files into document objects that can be processed.#Data cleaning
#3. Text splitting/chunking. chunking improves retival efficiency
#4. Each chunk is converted into vector embedding [Text -> vectors with semantic meaning]
#5. All vectors are stored in vector based storage DB
    #it stores(embeddings, original text and metadata)
#6. user interaction- query is conveterd to embeddings
#7. vectors db nperforms semantic similarity search. Find chinks most relevant
#8. retrivaer selectes top k chunks
#9. send the top k chunks to LLM along with query , LLM answers based on the context

##Advantages of chunking
# 1.Most of the LLMs have context windows - to preserve tokens , so they cannot take infinite tokens, 
# hence spiltting is needed
# 2. In Retrieval-Augmented Generation (RAG), we search through vector embeddings.If the document is too large:
#    embeddings become less precise. Smaller chunks allow the system to retrieve only the most relevant piece
#    of information instead of the whole document.  
#3. Fatser prcessing

### types of text splitters/chunking
# 1 character based defalt \n\n chunks =charcahters
# 2 Token based - no of tokens , chunks = tokens
# 3 Recursive character based[\n\n, \n, " ", ""]
# 4 semantic meaning based , 

# CHUNKING is the process of breaking a large piece of text (or data) into smaller, 
# manageable segments ("chunks") so it can be processed, stored, retrieved, or understood more effectively. 
# The goal is usually to balance two competing needs:
# 1. Enough context in each chunk so it makes sense on its own
# 2. Small enough size to fit model context windows, embedding limits, or processing constraints

## tokenization - breaking the sentence into tokens(tokens can be words. helps in text segmentation)
## chatgpt uses tiktoken library by openAI for tokenization

#The biggest problem is this 512 dimension embedding of our query
#is different from all the 1 lakh embeddings in our
#database, and we cannot search based on index like for RDS DB, 
#so we conduct a similarity search(cosine) with all the 1 lakh embeddings,which will cost O(n) time complexity

# draw lines betwen 1 lakh embeddings and draw line for query embedding and with which line the 
# angle is smallest is called cosine similarity

loader = TextLoader("document_loaders/notes.txt", encoding="utf-8")
documents = loader.load()

print(documents)
#print(documents[0].page_content)