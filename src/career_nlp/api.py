"""REST wrapper (FastAPI) around ``CareerNLP``.

    uvicorn career_nlp.api:app                 # run from base version/src
    # or
    python -m career_nlp serve --port 8000

Environment variables (see AssetConfig.from_env): CAREER_NLP_JOBS_PATH,
CAREER_NLP_CACHE_DIR, CAREER_NLP_MODEL_CACHE_DIR, CAREER_NLP_BACKEND,
CAREER_NLP_EMBEDDING_MODEL, CAREER_NLP_WARMUP.
API-level: CAREER_NLP_API_KEY (if set, every /v1 route except /v1/health needs
header ``X-API-Key``), CAREER_NLP_CORS_ORIGINS (comma separated; default none).
"""

from __future__ import annotations

import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import AssetConfig
from .engine import CareerNLP
from .errors import CVProcessingError, CareerNLPError
from .schemas import (
    AnalyzeRequest, AnalyzeResponse, ClassifyRequest, ClassifyResponse, CompareJobsRequest,
    CompareJobsResponse, CVResponse, EmbedRequest, EmbedResponse, ErrorBody, ErrorResponse,
    ExtractResponse, HealthResponse, InfoResponse, MatchRequest, MatchResponse, RecommendRequest, RecommendResponse,
    RetrieveRequest, RetrieveResponse, SimilarityRequest, SimilarityResponse,
    SummarizeRequest, SummarizeResponse, TextRequest,
)
from .version import ASSET_VERSION, SCHEMA_VERSION

log = logging.getLogger("career_nlp.api")

_ERRORS = {
    401: {"model": ErrorResponse, "description": "Missing/invalid API key"},
    422: {"model": ErrorResponse, "description": "Invalid input"},
    503: {"model": ErrorResponse, "description": "Model/data not available"},
}


def _error(status: int, code: str, message: str, request: Request, details=None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(
        code=code, message=message, details=details or {},
        request_id=getattr(request.state, "request_id", None)))
    return JSONResponse(status_code=status, content=body.model_dump())


def create_app(engine: Optional[CareerNLP] = None, config: Optional[AssetConfig] = None) -> FastAPI:
    config = config or AssetConfig.from_env()
    nlp = engine or CareerNLP(config)
    api_key = os.environ.get("CAREER_NLP_API_KEY")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if nlp.config.warmup_on_start:
            try:
                nlp.warmup()
            except CareerNLPError as exc:  # keep serving /health so the failure is visible
                log.error("Warm-up failed: %s", exc.message)
        yield

    app = FastAPI(
        title="Career NLP Asset", version=ASSET_VERSION, lifespan=lifespan,
        description="Reusable NLP component for Placement & Career Intelligence: extraction, "
                    "classification, embeddings, matching, recommendation and retrieval - "
                    "with structured output and confidence scores.",
    )
    app.state.nlp = nlp

    origins = [o.strip() for o in os.environ.get("CAREER_NLP_CORS_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    @app.exception_handler(CareerNLPError)
    async def _career_error(request: Request, exc: CareerNLPError):
        return _error(exc.http_status, exc.code, exc.message, request, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = {"errors": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]}
        return _error(422, "invalid_request", "Request body failed validation.", request, details)

    class _Unauthorized(CareerNLPError):
        code, http_status = "unauthorized", 401

    def auth(x_api_key: Optional[str] = Header(default=None)):
        if api_key and not (x_api_key and secrets.compare_digest(x_api_key, api_key)):
            raise _Unauthorized("Missing or invalid X-API-Key header.")

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(auth)], responses=_ERRORS)

    @app.get("/v1/health", response_model=HealthResponse, tags=["service"])
    def health():
        ready = nlp.is_ready
        st = nlp._state
        return HealthResponse(
            status="ok" if ready else "degraded", ready=ready, asset_version=ASSET_VERSION,
            schema_version=SCHEMA_VERSION,
            embedding_backend=st.backend if st else None, jobs_indexed=len(st.job_ids) if st else None)

    @v1.get("/info", response_model=InfoResponse, tags=["service"])
    def info():
        return nlp.info()

    @v1.post("/extract", response_model=ExtractResponse, tags=["nlp"])
    def extract(req: TextRequest):
        return nlp.extract(req.text)

    @v1.post("/classify", response_model=ClassifyResponse, tags=["nlp"])
    def classify(req: ClassifyRequest):
        return nlp.classify(req.text, req.top_k)

    @v1.post("/embed", response_model=EmbedResponse, tags=["nlp"])
    def embed(req: EmbedRequest):
        return nlp.embed(req.texts)

    @v1.post("/similarity", response_model=SimilarityResponse, tags=["nlp"])
    def similarity(req: SimilarityRequest):
        return nlp.similarity(req.text_a, req.text_b)

    @v1.post("/summarize", response_model=SummarizeResponse, tags=["nlp"])
    def summarize(req: SummarizeRequest):
        return nlp.summarize(req.text, req.max_sentences)

    @v1.post("/retrieve", response_model=RetrieveResponse, tags=["nlp"])
    def retrieve(req: RetrieveRequest):
        return nlp.retrieve(req.query, req.top_k)

    @v1.post("/match", response_model=MatchResponse, response_model_exclude_none=True, tags=["matching"])
    def match(req: MatchRequest):
        """JD PROVIDED: student text vs. one job description."""
        return nlp.match(req.student_text, req.job_text, req.include_details)

    @v1.post("/recommend", response_model=RecommendResponse, response_model_exclude_none=True, tags=["matching"])
    def recommend(req: RecommendRequest):
        """JD NOT PROVIDED: best job(s) from the full job dataset."""
        return nlp.recommend(req.student_text, req.top_k, req.include_details)

    @v1.post("/compare-jobs", response_model=CompareJobsResponse, response_model_exclude_none=True, tags=["matching"])
    def compare_jobs(req: CompareJobsRequest):
        """Compare multiple caller-supplied JDs for one student and rank the results."""
        return nlp.compare_jobs(req.student_text, req.job_descriptions, req.include_details)

    @v1.post("/analyze", response_model=AnalyzeResponse, response_model_exclude_none=True, tags=["matching"])
    def analyze(req: AnalyzeRequest):
        """Single entry point: runs match if job_text is given, else recommend."""
        return nlp.analyze(req.student_text, req.job_text, req.top_k, req.include_details)

    @v1.post("/cv/analyze", response_model=CVResponse, tags=["cv"])
    async def cv_analyze(file: UploadFile = File(...)):
        data = await file.read()
        return nlp.analyze_cv_bytes(data, file.filename or "resume")

    @v1.post("/extract-document", tags=["cv"])
    async def extract_document(file: UploadFile = File(...)):
        data = await file.read()
        filename = file.filename or "document"
        text = nlp._read_cv_text(data, filename)
        if not text.strip():
            raise CVProcessingError(
                "No text could be extracted. Scanned or image-only PDFs need OCR.",
                {"filename": filename},
            )
        return {"filename": os.path.basename(filename), "text": text}

    app.include_router(v1)
    return app


app = create_app()
