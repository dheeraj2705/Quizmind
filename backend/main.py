"""
main.py — FastAPI application for QuizMind AI.

Endpoints:
  POST   /upload                   Upload a study document
  POST   /generate-quiz            Generate quiz questions from stored material
  GET    /materials                List all uploaded materials
  DELETE /materials/{filename}     Remove a material and its embeddings
  POST   /score-answer             LLM-grade a short-answer response
  GET    /health                   Service health check
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Path, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from embeddings import EmbeddingManager
from file_parser import chunk_text, parse_file
from quiz_engine import QuizEngine, Question

# ---------------------------------------------------------------------------
# Application globals
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES = int(os.environ.get("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024

embedding_manager: EmbeddingManager | None = None
quiz_engine: QuizEngine | None = None


# ---------------------------------------------------------------------------
# Lifespan: initialise heavy models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_manager, quiz_engine

    print("[QuizMind] Loading embedding model and connecting to ChromaDB …")
    embedding_manager = EmbeddingManager()
    print("[QuizMind] Embedding manager ready.")

    try:
        quiz_engine = QuizEngine()
        print("[QuizMind] Quiz engine ready.")
    except EnvironmentError as exc:
        print(f"[QuizMind] WARNING: {exc}")

    yield

    print("[QuizMind] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

cors_origins_raw = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500",
)
cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]

app = FastAPI(
    title="QuizMind AI",
    description="AI-powered quiz generation from your study materials.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: safe manager access
# ---------------------------------------------------------------------------

def get_embedding_manager() -> EmbeddingManager:
    if embedding_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not ready. Please try again shortly.",
        )
    return embedding_manager


def get_quiz_engine() -> QuizEngine:
    if quiz_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Quiz engine not ready. Check that GROQ_API_KEY is set.",
        )
    return quiz_engine


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    success: bool
    filename: str
    chunks_stored: int
    message: str


class GenerateQuizRequest(BaseModel):
    topic: Optional[str] = None
    num_questions: int = Field(default=10, ge=5, le=20)
    difficulty: str = Field(default="medium")
    question_types: List[str] = Field(
        default=["mcq", "true_false", "short_answer"]
    )

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        allowed = {"easy", "medium", "hard"}
        if v.lower() not in allowed:
            raise ValueError(f"difficulty must be one of {allowed}")
        return v.lower()

    @field_validator("question_types")
    @classmethod
    def validate_question_types(cls, v: List[str]) -> List[str]:
        allowed = {"mcq", "true_false", "short_answer"}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Invalid question types: {invalid}. Allowed: {allowed}")
        if not v:
            raise ValueError("question_types must not be empty.")
        return v


class GenerateQuizResponse(BaseModel):
    questions: List[Dict]
    topic: str
    difficulty: str
    total_questions: int


class MaterialInfo(BaseModel):
    filename: str
    chunks: int
    uploaded_at: str


class MaterialsResponse(BaseModel):
    materials: List[MaterialInfo]


class ScoreAnswerRequest(BaseModel):
    question: Dict[str, Any]
    user_answer: str


class ScoreAnswerResponse(BaseModel):
    correct: bool
    score: float
    feedback: str


class HealthResponse(BaseModel):
    status: str
    chromadb_ready: bool
    embedding_model_loaded: bool
    quiz_engine_ready: bool
    timestamp: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_material(file: UploadFile = File(...)):
    """Parse an uploaded document, embed chunks, and store in ChromaDB."""
    mgr = get_embedding_manager()

    # Validate content type / extension
    allowed_extensions = {".pdf", ".docx", ".txt"}
    filename = file.filename or "upload"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Accepted: PDF, DOCX, TXT.",
        )

    # Read and validate size
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Parse file to text
    try:
        text = parse_file(file_bytes, filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File parsing failed: {exc}",
        )

    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text could be extracted from the file.",
        )

    # Chunk text
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text extraction produced no viable chunks.",
        )

    # Build metadata
    uploaded_at = datetime.now(timezone.utc).isoformat()
    metadata = [
        {
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "uploaded_at": uploaded_at,
        }
        for i in range(len(chunks))
    ]

    # Remove any previously stored version of the same file (upsert is fine
    # but a stale chunk count is confusing for the user).
    try:
        mgr.delete_material(filename)
    except Exception:
        pass  # First upload — nothing to delete

    stored = mgr.embed_and_store(chunks, metadata)

    return UploadResponse(
        success=True,
        filename=filename,
        chunks_stored=stored,
        message=f"Successfully processed '{filename}' into {stored} searchable chunks.",
    )


@app.post("/generate-quiz", response_model=GenerateQuizResponse, status_code=status.HTTP_200_OK)
async def generate_quiz(req: GenerateQuizRequest):
    """Retrieve relevant chunks from ChromaDB and generate a quiz via Groq."""
    mgr = get_embedding_manager()
    engine = get_quiz_engine()

    # Build semantic query from topic or use wildcard-style generic query
    query = req.topic if req.topic and req.topic.strip() else "key concepts main ideas definitions"

    # Optional: filter by filename/topic via metadata if topic matches a filename
    chunks = mgr.query_relevant_chunks(query, n_results=10)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No study material found. Please upload documents first.",
        )

    try:
        questions = engine.generate_quiz(
            context_chunks=chunks,
            num_questions=req.num_questions,
            difficulty=req.difficulty,
            question_types=req.question_types,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz generation failed: {exc}",
        )

    return GenerateQuizResponse(
        questions=[q.model_dump() for q in questions],
        topic=req.topic or "General",
        difficulty=req.difficulty,
        total_questions=len(questions),
    )


@app.get("/materials", response_model=MaterialsResponse, status_code=status.HTTP_200_OK)
async def list_materials():
    """Return a list of all uploaded materials with their chunk counts."""
    mgr = get_embedding_manager()
    raw = mgr.get_all_materials()
    materials = [
        MaterialInfo(
            filename=m["filename"],
            chunks=m["chunks"],
            uploaded_at=m.get("uploaded_at", ""),
        )
        for m in raw
    ]
    return MaterialsResponse(materials=materials)


@app.delete("/materials/{filename}", status_code=status.HTTP_200_OK)
async def delete_material(filename: str = Path(..., description="Filename to delete")):
    """Delete all stored chunks for a specific uploaded file."""
    mgr = get_embedding_manager()
    deleted = mgr.delete_material(filename)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No material found with filename '{filename}'.",
        )
    return {"success": True, "message": f"'{filename}' has been removed."}


@app.post("/score-answer", response_model=ScoreAnswerResponse, status_code=status.HTTP_200_OK)
async def score_answer(req: ScoreAnswerRequest):
    """Use Groq LLM to evaluate a short-answer quiz response."""
    engine = get_quiz_engine()
    try:
        result = engine.score_answer(req.question, req.user_answer)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {exc}",
        )
    return ScoreAnswerResponse(
        correct=result.correct,
        score=result.score,
        feedback=result.feedback,
    )


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Return the health status of the API and its dependencies."""
    chroma_ok = embedding_manager is not None and embedding_manager.is_ready()
    model_ok = embedding_manager is not None
    engine_ok = quiz_engine is not None

    return HealthResponse(
        status="ok" if (chroma_ok and model_ok) else "degraded",
        chromadb_ready=chroma_ok,
        embedding_model_loaded=model_ok,
        quiz_engine_ready=engine_ok,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
