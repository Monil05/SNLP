"""FastAPI wrapper for the Group 8 Week 3 core NLP pipeline.

Run from project root:
    uvicorn "base version.src.week3.api:app" --reload

Or from base version/src/week3:
    uvicorn api:app --reload
"""

from pathlib import Path
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

WEEK3_DIR = Path(__file__).resolve().parent
if str(WEEK3_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK3_DIR))

from nlp_v1 import V1NLP  # noqa: E402


app = FastAPI(
    title="Placement & Career Intelligence — NLP API",
    version="3.0.0",
    description="Week 3 core NLP pipeline for CV/job analysis, skill + semantic matching, job recommendation and retrieval.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

pipeline = V1NLP()


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SimilarityRequest(BaseModel):
    text_a: str = Field(..., min_length=1)
    text_b: str = Field(..., min_length=1)


class MatchRequest(BaseModel):
    student_text: str = Field(..., min_length=1)
    job_text: str = Field(..., min_length=1)


class RecommendRequest(BaseModel):
    student_text: str = Field(..., min_length=1)


class PipelineRequest(MatchRequest):
    top_k: int = Field(default=5, ge=1, le=20)


class RAGRequest(TextRequest):
    top_k: int = Field(default=5, ge=1, le=20)


class MatchResponse(BaseModel):
    match_score: float
    matched_skills: str
    missing_weak_skills: str
    recommended_learning: str


class RecommendResponse(BaseModel):
    recommended_job: str
    match_score: float
    matched_skills: str
    missing_weak_skills: str
    recommended_learning: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_version": "3.0",
        "components": [
            "ner",
            "classification",
            "embeddings",
            "semantic_similarity",
            "rag",
            "summarization",
            "information_extraction",
            "skill_and_semantic_matching",
            "job_recommendation",
        ],
    }


@app.post("/nlp/extract")
def extract(request: TextRequest):
    return pipeline.extract_entities(request.text)


@app.post("/nlp/classify")
def classify(request: TextRequest):
    return pipeline.classify_job(request.text)


@app.post("/nlp/embed")
def embed(request: TextRequest):
    return pipeline.embed([request.text])


@app.post("/nlp/similarity")
def similarity(request: SimilarityRequest):
    return pipeline.similarity(request.text_a, request.text_b)


@app.post("/nlp/summarize")
def summarize(request: TextRequest):
    return pipeline.summarize(request.text)


@app.post("/nlp/rag")
def rag(request: RAGRequest):
    return pipeline.rag(request.text, request.top_k)


@app.post("/nlp/match", response_model=MatchResponse)
def match(request: MatchRequest):
    """JD PROVIDED flow: a student profile matched against one specific job
    description. Returns only Match Score, Matched Skills, Missing/Weak
    Skills and Recommended Learning."""
    return pipeline.match_with_jd(request.student_text, request.job_text)


@app.post("/nlp/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """JD NOT PROVIDED flow: a student profile only. Searches/ranks the full
    cleaned job dataset and returns the single best-matching job."""
    return pipeline.recommend_job(request.student_text)


@app.post("/nlp/pipeline")
def full_pipeline(request: PipelineRequest):
    """Internal debug/demo view exposing every component (including the
    Week 2 baseline, kept for comparison). Not one of the two required
    user-facing flows — use /nlp/match and /nlp/recommend for those."""
    return pipeline.pipeline(request.student_text, request.job_text, request.top_k)


@app.post("/nlp/cv")
async def cv(file: UploadFile = File(...)):
    suffix = Path(file.filename or "resume").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX CVs are supported in Week 3.")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(data)
        temp_path = Path(temp.name)
    try:
        return pipeline.process_cv(temp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
