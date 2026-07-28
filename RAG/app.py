from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Mauli's Document Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Document Q&A Assistant")
st.caption("Ask questions about the document stored in your Chroma database.")


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the embedding model only once.

    Streamlit reruns the entire script after every interaction.
    st.cache_resource prevents the model from being reloaded
    every time the user asks a question.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


# ---------------------------------------------------------
# Load Chroma vector database
# ---------------------------------------------------------

@st.cache_resource
def load_vectorstore() -> Chroma:
    """
    Connect to the existing persistent Chroma database.
    """
    db_path = Path(__file__).resolve().parent / "chroma-db"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Chroma database was not found at: {db_path}"
        )

    embedding_model = load_embedding_model()

    return Chroma(
        collection_name="chromavectorstore",
        persist_directory=str(db_path),
        embedding_function=embedding_model,
    )


# ---------------------------------------------------------
# Load Mistral model
# ---------------------------------------------------------

@st.cache_resource
def load_llm() -> ChatMistralAI:
    """
    Create the Mistral chat model only once.
    """
    return ChatMistralAI(
        model="mistral-small-2603",
        temperature=0,
        max_retries=2,
    )


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use only the provided document context to answer the user's question.

Do not use outside knowledge.

If the answer is not available in the context, respond exactly with:

"I don't know. I am sorry, Mauli."

You may answer in a light and funny way, but the answer must remain accurate.
""",
        ),
        (
            "human",
            """
Document context:

{context}

User question:

{question}
""",
        ),
    ]
)


# ---------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello Mauli! Ask me something about your document.",
        }
    ]


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Retrieval settings")

    number_of_results = st.slider(
        "Number of chunks returned",
        min_value=1,
        max_value=10,
        value=3,
        help="This is the k value.",
    )

    candidate_chunks = st.slider(
        "Candidate chunks",
        min_value=number_of_results,
        max_value=50,
        value=10,
        help="This is fetch_k. MMR chooses the final chunks from these candidates.",
    )

    diversity_balance = st.slider(
        "Similarity vs. diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help=(
            "1.0 favors similarity. "
            "0.0 favors diversity. "
            "0.5 balances both."
        ),
    )

    show_sources = st.checkbox(
        "Show retrieved chunks",
        value=True,
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask me another document question.",
            }
        ]
        st.rerun()


# ---------------------------------------------------------
# Display existing chat messages
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources") and show_sources:
            with st.expander("View retrieved document chunks"):
                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    st.markdown(f"### Chunk {index}")
                    st.write(source["content"])

                    if source["metadata"]:
                        st.caption(
                            f"Metadata: {source['metadata']}"
                        )

                    st.divider()


# ---------------------------------------------------------
# Process a new question
# ---------------------------------------------------------

query = st.chat_input("Ask a question about your document...")

if query:
    # Store and display the user's question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Generate the assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                vectorstore = load_vectorstore()
                llm = load_llm()

                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": number_of_results,
                        "fetch_k": candidate_chunks,
                        "lambda_mult": diversity_balance,
                    },
                )

                results = retriever.invoke(query)

                if not results:
                    answer = "I don't know. I am sorry, Mauli."
                    sources = []

                else:
                    context = "\n\n".join(
                        document.page_content
                        for document in results
                    )

                    chain = prompt_template | llm

                    response = chain.invoke(
                        {
                            "context": context,
                            "question": query,
                        }
                    )

                    answer = response.content

                    sources = [
                        {
                            "content": document.page_content,
                            "metadata": document.metadata,
                        }
                        for document in results
                    ]

                st.markdown(answer)

                if sources and show_sources:
                    with st.expander("View retrieved document chunks"):
                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):
                            st.markdown(f"### Chunk {index}")
                            st.write(source["content"])

                            if source["metadata"]:
                                st.caption(
                                    f"Metadata: {source['metadata']}"
                                )

                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except FileNotFoundError as error:
                st.error(str(error))

            except Exception as error:
                st.error(
                    "Something went wrong while processing the question."
                )
                st.exception(error)