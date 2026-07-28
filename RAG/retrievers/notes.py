## when a query comes in , 
# 1> reriever searches the vector DB using any similarity search like below, like give TOP K results,
# which is also called  "RELEVANT CONTEXT":
   
# 2> The similar chunks(relevant context)+ query are sent to the LLM in the "PROMPT TEMPLATE"

# ----retriever --------------   
# A retriever takes a user query and returns the most relevant
# documents or chunks from a database.

# Now there are 2 types of retriever
# 1> by data source (Wikipedia, Arxiv, PubMed, etc.)
    # this can be used for searching through the specic sites. 
    # lets say Arxiv has all the resarch papers of data sciencie and u want to serach ""what is LLM" 
    # actross all the papers , then we can get the API KEY and search aross all Arxiv research papers 
    #SINCE THESE ARE EXTERNAL DATA SOURCES(DB/DATASOURCES), WE NEED API KEY. 
    # It WONT be across your own vector DB
    # ex: refer arxiv.py
# 2> By retrieval strategy (Similarity, MMR, MultiQuery, etc.) - mostly used in RAG
 # this retirever fetches across external sources 
        # a> similarity search
             # A.consine similarity
             # B.DOT
             # C.Eucladian
        
        # b> Max marginal Relevance(MMR). hence instead of retrieving similar documents,
            #  it retrieves based on below
            # Goal is to balance 2 things:
            #     1. Relevance to the query
            #     2. Diversity among Retrieved documents
 
 #WHATEVER U CAN INVOKE, U CAN CALL THEM RUNNABLES
 
 