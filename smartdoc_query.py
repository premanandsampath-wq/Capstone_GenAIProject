# =============================================================================
# Capstone Project: AI-Powered Enterprise Document Query System
# ILT PGP ge_B8
# Tools: Streamlit (UI) | ChromaDB (Vector Store) | Google Gemini (LLM)
# =============================================================================


# ── Task 1: Project Foundation ────────────────────────────────────────────────
import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") #or st.secrets.get("GOOGLE_API_KEY", "")



# ── All LangChain imports (correct paths for latest versions) ─────────────────
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# Task 2: User Interaction Layer — Streamlit Page Setup
# =============================================================================
st.set_page_config(page_title="Document QA Agent", page_icon="🤖", layout="wide")
st.title("AI Document Query Assistant")
st.caption("Upload documents and ask questions — powered by Google Gemini + ChromaDB")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =============================================================================
# Task 3: Document Ingestion — Load PDF, TXT, CSV, Excel
# =============================================================================
def load_document(file_path, file_name):
    File_ext = file_name.split(".")[-1].lower()

    if File_ext == "pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    elif File_ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
        pages = loader.load()
    elif File_ext == "csv":
        loader = CSVLoader(file_path, encoding="utf-8")
        pages = loader.load()
    elif File_ext in ["xlsx", "xls"]:
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        pages = [Document(page_content=text, metadata={"source": file_name})]
    else:
        st.error(f"Unsupported file type: {File_ext}")
        return []

    return pages


# =============================================================================
# Task 4: Chunking — Prepare data for semantic search
# =============================================================================
def split_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)
    return chunks


# =============================================================================
# Task 5: Vector Store — ChromaDB + Google Embeddings
# =============================================================================
def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        # api_version="v1"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="documents",
    )
    return vector_store


# =============================================================================
# Task 6: Retrieval — Similarity search
# =============================================================================
def get_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return retriever


# =============================================================================
# Task 7: RAG Pipeline — Retrieval + Gemini LLM
# =============================================================================
def build_rag_chain(retriever):
    prompt_template = """
You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    # This returns BOTH answer and source documents
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return rag_chain, llm, retriever


# =============================================================================
# Task 8: Agent-Based Reasoning (ReAct Agent with Tool)
# =============================================================================
def build_agent(llm, retriever):
    """
    Creates a ReAct-style AgentExecutor.
    The LLM decides WHEN and WHETHER to call the DocumentSearch tool.
    """

    # 1. Wrap the retriever as a named Tool the agent can choose to call
    def search_documents(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join([d.page_content for d in docs])

    retriever_tool = Tool(
        name="DocumentSearch",
        func=search_documents,
        description=(
            "Use this tool to search through uploaded documents and retrieve "
            "relevant information to answer the user's question. "
            "Input should be a clear search query."
        ),
    )

    tools = [retriever_tool]

    # 2. Pull the standard ReAct prompt from LangChain Hub
    react_prompt = hub.pull("hwchase17/react")

    # 3. Create the ReAct agent — LLM reasons: Thought → Action → Observation → Answer
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt,
    )

    # 4. Wrap in AgentExecutor which runs the reasoning loop
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,               # prints reasoning steps to terminal
        handle_parsing_errors=True,
        max_iterations=5,           # prevents infinite loops
    )

    return agent_executor


def agent_answer(agent_executor, question):
    """
    Invokes the ReAct agent. The agent plans, searches, and generates.
    """
    if len(question.strip()) < 3:
        return "Please ask a more specific question.", []

    result = agent_executor.invoke({"input": question})
    answer = result.get("output", "No answer generated.")
    return answer, []


# =============================================================================
# Task 9: Safety Controls
# =============================================================================
def validate_question(question):
    if not question or not question.strip():
        return False, " Please enter a question."
    if len(question.strip()) <= 3:
        return False, " Question too short. Please be more specific."
    if len(question) > 500:
        return False, " Question too long. Keep it under 500 characters."

    banned = ["ignore instructions", "forget everything", "jailbreak"]
    if any(b in question.lower() for b in banned):
        return False, " Invalid input detected."

    return True, ""


def safe_answer(agent_executor, question):
    try:
        return agent_answer(agent_executor, question)
    except Exception as e:
        return f" Error: {str(e)}", []


# =============================================================================
# Task 10: Deploy — Streamlit UI
# =============================================================================

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, TXT, CSV, Excel)",
        type=["pdf", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if st.button(" Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            all_chunks = []
            with st.spinner("Reading and indexing documents..."):
                for file in uploaded_files:
                    suffix = "." + file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    pages = load_document(tmp_path, file.name)
                    if pages:
                        chunks = split_into_chunks(pages)
                        all_chunks.extend(chunks)
                        st.success(f" {file.name} → {len(chunks)} chunks")

                if all_chunks:
                    vector_store   = build_vector_store(all_chunks)
                    retriever      = get_retriever(vector_store)
                    rag_chain, llm, retriever = build_rag_chain(retriever)  # unpacks 3 values
                    agent_executor = build_agent(llm, retriever)            # builds ReAct agent

                    st.session_state.rag_chain      = rag_chain             # kept for reference
                    st.session_state.agent_executor = agent_executor        # agent used for chat
                    st.success("✅ Ready! Ask your questions →")

    st.markdown("---")
    if st.button(" Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── Chat ──────────────────────────────────────────────────────────────────────
st.markdown("###  Ask a Question About Your Documents")

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

question = st.chat_input("Type your question here...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append(("user", question))

    valid, err_msg = validate_question(question)

    if not valid:
        with st.chat_message("assistant"):
            st.warning(err_msg)
        st.session_state.chat_history.append(("assistant", err_msg))

    elif "rag_chain" not in st.session_state:
        msg = " Please upload and process documents first using the sidebar."
        with st.chat_message("assistant"):
            st.warning(msg)
        st.session_state.chat_history.append(("assistant", msg))

    else:
        with st.spinner(" Executing..."):
            answer, sources = safe_answer(st.session_state.agent_executor, question)

        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                with st.expander(" View Sources"):
                    for i, doc in enumerate(sources, 1):
                        src = doc.metadata.get("source", "Unknown")
                        st.caption(f"Source {i}: {src}")
                        st.text(doc.page_content[:300] + "...")

        st.session_state.chat_history.append(("assistant", answer))
#==========================================================================================