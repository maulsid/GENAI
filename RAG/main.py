from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

#loader = TextLoader("document_loaders/notes.txt", encoding="utf-8")
loader = PyPDFLoader("document_loaders/GRU.pdf")
documents = loader.load()
content1 = documents[0].page_content
#print(documents[0].page_content)

llm = ChatMistralAI( 
    model="mistral-small-2603",
    temperature=0,
    max_retries=2,
)

template = ChatPromptTemplate.from_messages(
        [
            ("system",  "You are a helpful funny assistant, summarise the document"),
            ("human", "{content}"),
        ]
    )
finalstr = template.format(content= content1)

#print(documents)
ai_msg = llm.invoke(finalstr)
print(ai_msg.content)