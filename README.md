# 🔍 Local File Finder AI

> **Search your Windows files using natural language — locally, privately, and safely.**

Local AI-powered file search for Windows that understands messy **English, Hindi, and Hinglish** queries using **Gemma 4 running locally through Ollama**.

Instead of manually browsing folders, simply ask:

> 🗣️ `mera aadhar card E drive mein find karo`

The system understands the request, searches the real filesystem, ranks the matching files, and returns **only real paths discovered from the filesystem**.

---

## ✨ Features

- 🧠 **Natural-language search** using local Gemma 4
- 🇮🇳 **Hinglish / English query understanding**
- 🔒 **Privacy-first** — no cloud AI calls required
- 🛡️ **Filesystem sandboxing** using `ALLOWED_ROOTS`
- 🚫 **Path traversal protection**
- 🎯 **Grounded LLM ranking** — hallucinated paths are rejected
- ⚡ **SQLite-based local file index**
- 🔄 **LLM fallback** when Ollama is unavailable
- 🚀 **FastAPI backend**
- 🎨 **Streamlit desktop-style UI**
- 🧩 **LangGraph workflow orchestration**
- 🔗 **LangChain structured LLM integration**
- 🧪 **Automated API, filesystem, and security tests**
- 📂 **Open matching files/folders directly from the UI**

---

# 🏗️ Architecture

```text
                    👤 User
                       │
                       ▼
              ┌─────────────────┐
              │    Streamlit    │
              │       UI        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     FastAPI     │
              │   POST /search  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    LangGraph    │
              │   Orchestrator  │
              └────────┬────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      🧠 Understand            🔎 Search
       Query (Gemma)       Filesystem Tools
             │                   │
             └─────────┬─────────┘
                       ▼
                 🧠 Rank Results
                    (Gemma)
                       │
                       ▼
              🎯 Grounded Results
                       │
                       ▼
                 Streamlit UI
```

### 🔄 LangGraph Workflow

```text
START
  │
  ▼
Understand Query
  │
  ├── No filesystem search needed
  │          │
  │          ▼
  │         Rank
  │
  └── Search required
             │
             ▼
           Search
             │
             ▼
            Rank
             │
             ▼
            END
```

---

# 🧠 How It Works

### 1️⃣ Understand

Gemma receives the user's natural-language request and converts it into a structured search plan.

For example:

```text
"mera aadhar card E drive mein find karo"
```

becomes conceptually:

```text
needs_search = true
location = E:\
search_terms = "aadhar card"
```

The LLM is **not trusted blindly**.

Any location returned by the model is validated against the configured allowed roots.

---

### 2️⃣ Search

The Python filesystem layer performs the actual search.

The filesystem tools are the only components that directly access the user's files.

Available tools:

```text
🔎 search_files()
📁 list_directory()
📄 get_file_info()
```

All filesystem operations are **read-only**.

---

### 3️⃣ Rank

The filesystem produces candidate files using deterministic matching.

Gemma then receives a limited shortlist of candidates and ranks them according to the original query.

The model must select paths from the provided candidate set.

```text
User Query
     │
     ▼
Candidate Files
     │
     ▼
Gemma Ranking
     │
     ▼
Real Existing Paths
```

If Gemma produces a path that was not present in the candidate list, it is rejected.

This prevents fabricated file paths from reaching the user.

---

# 🔒 Security Design

Security is a core part of the project.

## 🛡️ Allowed Roots

Filesystem access is restricted using:

```env
ALLOWED_ROOTS=E:\
```

Only paths inside the configured roots can be accessed.

---

## 🚫 Path Traversal Protection

Attempts such as:

```text
..\..\Windows
C:\Windows\System32
```

are rejected.

Paths are validated through:

```python
resolve_and_validate()
```

before filesystem operations are performed.

---

## 🔐 No Arbitrary Shell Access

Gemma does **not** receive shell access.

There is no:

```text
os.system()
subprocess
shell execution
```

for filesystem searching.

The LLM communicates its intent, while controlled Python functions perform the actual disk operations.

---

## 🚫 Hallucinated Path Protection

The ranking stage uses a grounding rule:

```text
LLM can only select from actual candidate paths.
```

Any path outside the candidate set is discarded.

---

# 🧩 Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Core application logic |
| 🧠 Gemma 4 | Local language understanding & ranking |
| 🦙 Ollama | Local LLM runtime |
| 🔗 LangChain | LLM integration & structured output |
| 🔄 LangGraph | Search workflow orchestration |
| ⚡ FastAPI | Backend REST API |
| 🎨 Streamlit | User interface |
| 🗃️ SQLite | Local file indexing |
| 📦 Pydantic | Data validation |
| 🧪 Pytest | Automated testing |

---

# 📁 Project Structure

```text
File_Search/
│
├── app/
│   ├── agent/
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── prompts.py
│   │
│   ├── schemas/
│   │   └── search.py
│   │
│   ├── services/
│   │   ├── api_client.py
│   │   └── ollama.py
│   │
│   ├── tools/
│   │   └── filesystem.py
│   │
│   ├── config.py
│   ├── main.py
│   └── streamlit_app.py
│
├── tests/
│   ├── test_api.py
│   ├── test_filesystem.py
│   └── test_security.py
│
├── .env.example
├── requirements.txt
└── README.md
```

---

# 🚀 Setup

## 1️⃣ Clone the Repository

```powershell
git clone https://github.com/Chittransh-89/File_Search.git
cd File_Search
```

---

## 2️⃣ Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3️⃣ Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment

Create your `.env` file:

```powershell
copy .env.example .env
```

Example configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e4b

ALLOWED_ROOTS=E:\

MAX_CANDIDATES=10
```

---

# 🦙 Ollama Setup

Make sure Ollama is installed and running.

Pull the model:

```powershell
ollama pull gemma4:e4b
```

You can verify the model with:

```powershell
ollama list
```

---

# ▶️ Run the Application

## 1️⃣ Start FastAPI

From the project root:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## 2️⃣ Start Streamlit

Open another terminal:

```powershell
streamlit run app/streamlit_app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open it in your browser.

---

# 🔎 Example Queries

The application can handle natural-language requests such as:

```text
mera aadhar card E drive mein find karo
```

```text
meri GitHub wali file dhoondo
```

```text
resume analyzer project ki files dhoondo
```

```text
E drive mein PDFs dhoondo
```

```text
college ki DBMS wali files dhoondo
```

```text
my python project files find karo
```

The goal is to describe **what you want**, rather than manually navigate through folders.

---

# 🎨 Streamlit UI

The UI provides:

- 🔍 Natural-language search
- 📍 Search location
- 🟢 FastAPI / Ollama / Gemma status
- 🕘 Recent search history
- 🎯 Best match
- 📊 Match score
- 💡 Match explanation
- 📂 Open folder
- 📄 Open file

The Streamlit layer does not implement the search logic.

It communicates with the FastAPI backend through:

```text
app/services/api_client.py
```

---

# ⚡ Local File Index

The application includes a SQLite-based local index.

You can build or rebuild the index through the API:

```powershell
curl -X POST http://127.0.0.1:8000/index
```

Or directly:

```powershell
python -c "from app.tools.filesystem import build_index; print(build_index('E:\\'))"
```

The index can speed up repeated searches and provides a foundation for future semantic retrieval.

---

# 🌐 API

## Health Check

```http
GET /health
```

Example:

```powershell
curl http://127.0.0.1:8000/health
```

---

## Search

```http
POST /search
```

Example:

```powershell
curl -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query":"mera aadhar card E drive mein find karo"}'
```

---

## Rebuild Index

```http
POST /index
```

Example:

```powershell
curl -X POST http://127.0.0.1:8000/index
```

---

# 🧪 Testing

Run the complete test suite:

```powershell
pytest -q
```

Tests cover:

### 🔎 Filesystem

- Real file discovery
- Extension filtering
- Empty search results
- Directory listing
- File metadata

### 🔐 Security

- Absolute path escapes
- Directory traversal
- Outside-root searches
- Outside-root file information
- Read-only filesystem behavior

### 🌐 API

- Real paths returned
- No-result responses
- Empty query validation
- Health endpoint

---

# 🧠 Fallback Behavior

The application does not completely break when the LLM is unavailable.

If Gemma/Ollama fails during query understanding:

```text
Gemma unavailable
       ↓
Fallback parser
       ↓
Generic token extraction
       ↓
Filesystem search
```

If Gemma fails during ranking:

```text
Gemma ranking unavailable
       ↓
Filesystem scores
       ↓
Fallback results
```

This keeps the core search functionality usable even without successful LLM generation.

---

# 🎯 Design Philosophy

This project deliberately separates **AI reasoning** from **filesystem control**.

```text
🧠 LLM
Understands language
       │
       ▼
📋 Structured Search Plan
       │
       ▼
🐍 Python
Controls filesystem access
       │
       ▼
📁 Real Files
       │
       ▼
📊 Candidate Ranking
       │
       ▼
🧠 LLM
Ranks grounded candidates
```

The LLM acts as the **brain**, while controlled Python filesystem functions act as the **hands**.

The model never gets unrestricted access to the operating system.

---

# 🚧 Milestone 1

Current milestone focuses on building a reliable local foundation:

- ✅ Natural-language file search
- ✅ Hinglish / English queries
- ✅ Local Gemma inference
- ✅ Ollama integration
- ✅ LangChain structured output
- ✅ LangGraph workflow
- ✅ Controlled filesystem tools
- ✅ Path validation
- ✅ Read-only filesystem access
- ✅ Candidate ranking
- ✅ Hallucinated-path filtering
- ✅ SQLite indexing
- ✅ FastAPI backend
- ✅ Streamlit UI
- ✅ Automated tests

---

# 🔮 Future Improvements

Possible future milestones:

- 🔎 Semantic search with embeddings
- 🧠 Vector retrieval
- 📄 Content-aware document search
- 🖼️ OCR support
- 📑 PDF/document content indexing
- ⚡ Faster incremental indexing
- 🎯 Improved ranking
- 🗂️ Advanced folder-aware retrieval
- 🤖 More intelligent query planning

These are intentionally **not part of Milestone 1**.

---

# 🔐 Privacy

This project is designed around local processing.

```text
User Query
     ↓
Local FastAPI
     ↓
Local LangGraph
     ↓
Local Ollama
     ↓
Local Filesystem
```

No external AI API is required.

Your queries and filesystem paths stay on your machine during normal local operation.

---

# 👨‍💻 Author

**Chittransh Verma**

Computer Science Engineering Student  
Interested in AI Engineering, GenAI, RAG, Backend Systems & Intelligent Applications.

---

⭐ If you find the project interesting, consider giving the repository a star!
