# QuizMind AI

> **Upload. Learn. Master.**  
> An AI-powered quiz application that transforms your study materials
> into adaptive quizzes using open-source models via Groq.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup](#quick-setup)
3. [Getting a Groq API Key](#getting-a-groq-api-key)
4. [How to Use](#how-to-use)
5. [Architecture](#architecture)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | `python --version` |
| pip | latest | `pip install --upgrade pip` |
| Groq API key | — | Free at [console.groq.com](https://console.groq.com) |
| (Optional) Live Server / VS Code Extension | — | For serving the frontend |

---

## Quick Setup

```bash
# 1. Clone or unzip the project
cd quizmind

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r backend/requirements.txt

# 4. Copy the example env file and add your Groq API key
copy .env.example backend\.env       # Windows
# cp .env.example backend/.env       # macOS / Linux

# Edit backend/.env and replace the placeholder:
# GROQ_API_KEY=gsk_...your_actual_key...

# 5. Start the backend API
cd backend
python main.py
# The API will be running at http://localhost:8000
# Interactive docs at  http://localhost:8000/docs

# 6. Open the frontend
# Simply open frontend/index.html in your browser, OR
# serve it with any static server, e.g. Python's built-in:
cd ../frontend
python -m http.server 5500
# Then visit http://localhost:5500
```

---

## Getting a Groq API Key

Groq provides **free API access** to open-source LLMs including LLaMA 3.3.

1. Go to **[console.groq.com](https://console.groq.com)**
2. Create a free account (no credit card required for the free tier)
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_…`)
5. Paste it into `backend/.env` as `GROQ_API_KEY=gsk_...`

The free tier generously supports development and light usage.

---

## How to Use

### Step 1 — Upload Study Material

1. Click **Upload Material** or navigate to the **Upload** tab
2. Drag-and-drop or click to browse for a **PDF**, **DOCX**, or **TXT** file
3. The backend will:
   - Extract text from the document
   - Split it into ~500-word chunks with 50-word overlap
   - Generate semantic embeddings using `all-MiniLM-L6-v2`
   - Store everything in a local ChromaDB vector store
4. You'll see the material listed with its chunk count

Upload as many documents as you like — they're all indexed together.

### Step 2 — Configure Your Quiz

1. Click **Quiz** or **Take a Quiz**
2. Set options:
   - **Number of Questions**: 5–20 (slider)
   - **Difficulty**: Easy / Medium / Hard
   - **Question Types**: Multiple Choice, True/False, Short Answer
   - **Topic Filter** (optional): focus the AI on a specific subject
3. Click **Generate Quiz**

### Step 3 — Take the Quiz

- **MCQ**: Click an option card (or press 1/2/3/4 or A/B/C/D)
- **True/False**: Click the True or False button
- **Short Answer**: Type your response, then click "Submit Answer"
- Use **← Back** / **Next →** or the dot navigator to move between questions
- Press **Enter** to advance, **Arrow keys** to navigate

### Step 4 — Review Results

- See your **score** as a percentage with an animated ring
- Receive a **grade badge**: Excellent / Good / Needs Review
- Click any question to expand and see your answer vs. the correct one,
  plus the AI's explanation

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser / Frontend                   │
│               frontend/index.html (vanilla JS)          │
│  Upload → Setup → Quiz Taking → Results                 │
└───────────────────────┬─────────────────────────────────┘
                        │  REST API (fetch)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 FastAPI Backend (port 8000)              │
│                    backend/main.py                      │
│                                                         │
│  POST /upload          → file_parser.py → embeddings.py │
│  POST /generate-quiz   → embeddings.py  → quiz_engine.py│
│  GET  /materials       → embeddings.py                  │
│  DELETE /materials/:f  → embeddings.py                  │
│  POST /score-answer    → quiz_engine.py                 │
│  GET  /health          → status check                   │
└────────┬────────────────────────┬───────────────────────┘
         │                        │
         ▼                        ▼
┌────────────────┐    ┌──────────────────────────────────┐
│  ChromaDB      │    │          Groq API                │
│  (local disk)  │    │  model: llama-3.3-70b-versatile  │
│  ./chroma_db   │    │  quiz generation + scoring       │
│                │    └──────────────────────────────────┘
│  Embeddings:   │
│  all-MiniLM    │
│  -L6-v2        │
└────────────────┘
```

### File Roles

| File | Responsibility |
|---|---|
| `backend/main.py` | FastAPI routes, validation, orchestration |
| `backend/embeddings.py` | SentenceTransformer + ChromaDB CRUD |
| `backend/quiz_engine.py` | Groq API calls, JSON parsing, scoring |
| `backend/file_parser.py` | PDF/DOCX/TXT parsing, text chunking |
| `frontend/index.html` | Complete single-file SPA |

---

## Troubleshooting

### `GROQ_API_KEY not set` error
Make sure `backend/.env` exists and contains `GROQ_API_KEY=gsk_...`.
Run the server from the `backend/` directory so `python-dotenv` finds the `.env` file.

### `No study material found` when generating quiz
Upload at least one document before generating a quiz.

### ChromaDB version conflicts
If you see errors related to ChromaDB, try:
```bash
pip install --upgrade chromadb
```
ChromaDB >= 0.4.22 is required for the `PersistentClient` API used here.

### Sentence-transformers takes long to start
The first run downloads the `all-MiniLM-L6-v2` model (~90 MB). This is cached
locally after the first download at `~/.cache/huggingface/`.

### CORS errors in browser
Ensure the `CORS_ORIGINS` variable in `backend/.env` includes the origin
you're using to serve the frontend. For local file:// access, also add
an empty string or use a local server on port 5500:
```
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500
```

### PDF text not extracted correctly
Some PDFs are image-based (scanned). QuizMind requires text-embedded PDFs.
Consider using OCR tools like `ocrmypdf` to pre-process scanned documents.

### PyTorch / torch install takes too long
For CPU-only machines, install the lighter CPU build first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Port 8000 already in use
Either stop the conflicting process or change the port in `main.py`:
```python
uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
```
Then update `const API_URL = "http://localhost:8001"` at the top of `frontend/index.html`.

---

## License

MIT — free to use, modify, and distribute.
