#  AI-Powered Enterprise Document Query System
***********************************************************************************************************************************
> **Capstone Project — ILT PGP ge_B8**  
> A conversational document Q&A assistant powered by Google Gemini, ChromaDB, and LangChain — with an interactive Streamlit UI.
***********************************************************************************************************************************
## Overview

This application allows users to upload enterprise documents (PDF, TXT, CSV, Excel) and ask natural language questions about their contents. It uses a ReAct Agent that intelligently decides when to retrieve document context, providing grounded, accurate answers.

***********************************************************************************************************************************

## Tech Stack

| Layer | Tool |
|---|---|
| **UI** | [Streamlit](https://streamlit.io/) |
| **LLM** | Google Gemini (`gemini-2.5-flash`) via LangChain |
| **Embeddings** | Google Generative AI Embeddings (`gemini-embedding-001`) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) |
| **Agent Framework** | LangChain ReAct Agent |
| **Document Loaders** | LangChain Community (PDF, TXT, CSV, Excel) |

***********************************************************************************************************************************
## Project Structure


├── app.py                  # Main Streamlit application
├── chroma_db/              # Persisted ChromaDB vector store (auto-created)
├── .env                    # API key configuration (not committed)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

***********************************************************************************************************************************

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <project-folder>
```
***********************************************************************************************************************************
### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```
***********************************************************************************************************************************
### 3. Install dependencies

```bash
pip install -r requirements.txt
```
***********************************************************************************************************************************
### 4. Configure your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```
***********************************************************************************************************************************
> You can obtain a Google API key from [Google AI Studio](https://aistudio.google.com/).

### 5. Run the application

```bash
streamlit run app.py
```
***********************************************************************************************************************************
## 📋 Requirements

Below are the key Python packages required. Add these to your `requirements.txt`:

```
streamlit
pandas
python-dotenv
langchain
langchain-google-genai
langchain-community
langchain-text-splitters
langchain-core
langchainhub
chromadb
pypdf
openpyxl
***********************************************************************************************************************************
## How to Use

1. **Launch** the app with `streamlit run app.py`.
2. **Upload documents** using the sidebar — supported formats: `.pdf`, `.txt`, `.csv`, `.xlsx`, `.xls`.
3. Click **"Process Documents"** to chunk, embed, and index them into ChromaDB.
4. **Type a question** in the chat input at the bottom of the page.
5. The ReAct Agent will search the documents and return a grounded answer.
---
***********************************************************************************************************************************

## Architecture & Task Breakdown

### Task 1 — Project Foundation
Loads environment variables and initialises the Google API key from `.env`.

### Task 2 — Streamlit Page Setup
Configures the Streamlit page (title, layout, session state for chat history).

### Task 3 — Document Ingestion
Supports loading of:
- **PDF** via `PyPDFLoader`
- **TXT** via `TextLoader`
- **CSV** via `CSVLoader`
- **Excel (.xlsx/.xls)** via `pandas` → converted to a `Document` object

### Task 4 — Chunking
Uses `RecursiveCharacterTextSplitter` with:
- `chunk_size = 500`
- `chunk_overlap = 50`

### Task 5 — Vector Store
Embeds chunks using **Google Generative AI Embeddings** and stores them in a local **ChromaDB** instance persisted to `./chroma_db`.

### Task 6 — Retrieval
Creates a similarity-search retriever returning the top **3** most relevant chunks per query.

### Task 7 — RAG Pipeline
Combines the retriever with a **PromptTemplate** and `ChatGoogleGenerativeAI` (`gemini-2.5-flash`, temperature 0.2) using LangChain's `RunnablePassthrough` pipeline.

### Task 8 — ReAct Agent
Wraps the retriever as a `DocumentSearch` tool and builds a **ReAct AgentExecutor** using `hwchase17/react` from LangChain Hub. The agent reasons through Thought → Action → Observation → Answer cycles, with a maximum of 5 iterations.

### Task 9 — Safety Controls
Validates user input before processing:
- Rejects empty or very short questions (≤ 3 characters)
- Rejects questions over 500 characters
- Blocks known prompt injection patterns (e.g. `"jailbreak"`, `"ignore instructions"`)
- Wraps execution in `try/except` for graceful error handling

### Task 10 — Streamlit UI
- **Sidebar**: file uploader and document processing button
- **Chat interface**: full chat history display, chat input, and expandable source viewer


## Safety & Limitations

- Answers are grounded in uploaded documents only — the model is instructed to say *"I don't know based on the documents"* if context is insufficient.
- Prompt injection patterns are filtered before reaching the LLM.
- Agent iterations are capped at 5 to prevent runaway loops.

**********************************************************************************************

