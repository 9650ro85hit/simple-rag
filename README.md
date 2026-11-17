# RAG Chatbot — AI Knowledge Workspace

An AI-powered chatbot that lets you interact with your own documents using Retrieval-Augmented Generation (RAG). Upload documents, ask questions, and get context-aware answers grounded in your uploaded content. When a question requires information beyond the uploaded documents, the system can use web search.

The application also supports general-purpose conversations, so you can use it both as a personal AI assistant and as a document-based knowledge assistant.

## ✨ Features

### 📚 Project-Based RAG Chat
- Create separate projects for different knowledge bases.
- Upload documents to a project.
- Ask questions based on the uploaded documents.
- Keep conversations organized by project.
- Maintain separate chat histories.

### 🔍 Advanced RAG Pipeline
- **Document ingestion:** Load and process uploaded documents.
- **Text chunking:** Split documents into smaller chunks for retrieval.
- **Vector embeddings:** Convert text chunks into numerical representations.
- **Vector storage:** Store embeddings using FAISS.
- **Query rewriting:** Improve user queries before retrieval.
- **Hybrid retrieval:** Combine semantic search and keyword-based retrieval.
- **RRF fusion:** Merge rankings from different retrieval methods.
- **Reranking:** Improve the relevance of retrieved chunks.
- **Relevance evaluation:** Evaluate whether retrieved content is relevant to the question.
- **Context-aware generation:** Generate answers using retrieved document context.

### 🌐 Web Search Fallback
When the uploaded documents do not provide sufficient information, the system can use web search to retrieve additional information and generate an answer.

### 💬 General Chatbot
- Chat without creating a project.
- Use the application for general-purpose questions.
- Keep general conversations separate from project-based conversations.

### 🧠 Conversation Memory
- Preserve conversation history across sessions.
- Use LangGraph checkpointing with SQLite for conversation persistence.
- Maintain conversational context for follow-up questions.
- Support conversation summarization to manage longer conversations.

### 🖥️ User Interface
- Project and general-chat organization.
- Create, rename, and delete projects and conversations.
- Separate chat histories.
- Light and dark themes.
- Streaming responses for a more interactive experience.

---

## 🏗️ Architecture

The application consists of a frontend, backend API, RAG pipeline, vector database, and persistent conversation storage.

```text
                  User
                   |
                   v
             Web Frontend
                   |
                   v
              FastAPI API
                   |
                   v
             Chat Mode Router
              /           \
             /             \
       Project Chat      General Chat
            |                 |
            v                 v
       LangGraph RAG       LLM
         Pipeline
            |
            v
       Query Rewriting
            |
            v
       Hybrid Retrieval
        /           \
   Dense Search   BM25 Search
        \           /
         \         /
          RRF Fusion
              |
              v
          Reranking
              |
              v
       Relevance Evaluation
              |
       +------+------+
       |             |
   Relevant      Insufficient
       |             |
       v             v
  Document-based   Web Search
    Generation         |
       |               |
       +-------+-------+
               |
               v
          Final Answer
               |
               v
        Stream to Frontend
```

### Document Ingestion Flow

```text
       Uploaded Documents
               |
               v
        Document Loading
               |
               v
          Text Splitting
               |
               v
        Text Embeddings
               |
               v
          FAISS Index
               |
               v
        Project Knowledge Base
```

### Technologies Used

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| RAG orchestration | LangGraph |
| LLM integration | LangChain |
| Vector database | FAISS |
| Embeddings | Sentence Transformers |
| Keyword retrieval | BM25 |
| Web search | Tavily |
| Conversation persistence | SQLite, LangGraph Checkpointing |
| Frontend | JavaScript, HTML, CSS |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Backend Setup

Create and activate a Python virtual environment.

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Configure your required API keys and environment variables in a `.env` file, according to your backend configuration.

Do not commit your `.env` file or API keys to GitHub.

### 3. Start the Backend

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at:

```text
http://localhost:8000
```

### 4. Frontend Setup

Open another terminal and navigate to the frontend directory.

```bash
cd rag-chatbot/rag-chatbot
```

Install the dependencies:

```bash
npm install
```

Start the frontend:

```bash
npm start
```

Open the local URL displayed in your terminal to access the application.

---

## 📖 How to Use

### 1. Create a Project
- Open the application.
- Create a new project.
- Give your project a name.
- Open the project to start working with its knowledge base.

### 2. Upload Documents
- Upload the documents you want to use as your knowledge base.
- The backend processes the documents, splits their content into chunks, generates embeddings, and stores them in the vector database.

### 3. Ask Questions
- Open a conversation inside your project.
- Enter a question about your uploaded documents.
- The RAG pipeline processes the question, retrieves relevant content, evaluates it, and generates an answer.

### 4. Continue the Conversation
- Ask follow-up questions.
- The application maintains conversation history to support contextual conversations.

### 5. Use General Chat
- Select General Chat to start a conversation without using a project knowledge base.
- Ask general questions and use the application as a regular AI assistant.

---

## 📁 Project Structure

The repository is organized around the backend RAG implementation and frontend application.

```text
RAG/
│
├── api.py                  # FastAPI backend and API endpoints
├── rag_graph.py            # LangGraph RAG workflow
├── rag_storage.py          # Document processing and vector storage
├── rag_chat_app.py         # Streamlit application (if retained)
│
├── rag-chatbot/
│   └── rag-chatbot/
│       ├── package.json    # Frontend dependencies and scripts
│       ├── package-lock.json
│       └── ...             # Frontend application files
│
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── .gitignore
└── README.md
```

*The structure above is illustrative; adjust it to match the files in your repository.*

---

## 🔐 Security

- Keep API keys and credentials in environment variables.
- Never commit `.env` files.
- Avoid uploading private or confidential documents to a public repository.
- Review generated files, local databases, and uploaded documents before committing.

---

## 🎯 Project Goal

The goal of this project is to build a practical AI knowledge assistant that combines document-based retrieval, semantic and keyword search, relevance evaluation, web search, and conversational memory.

It is also an ongoing exploration of building production-oriented RAG systems using LangGraph and modern LLM application architecture.

## 🤝 Contributions

Suggestions, bug reports, and contributions are welcome. Feel free to open an issue or submit a pull request.
