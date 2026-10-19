"""Structured input/output contract (Pydantic v2).

These models are the public contract of the asset. They are used identically
by the Python API (``CareerNLP`` methods return them), the CLI (``--json``) and
the REST layer (request bodies / ``response_model``). JSON Schemas are exported
to ``docs/schemas/`` by ``examples/generate_docs.py``.

Conventions
-----------
* Scores named ``*_percent`` and ``match_score`` are on a 0-100 scale.
* Probabilities / confidence ``score`` values are on a 0-1 scale.
* Skill collections are JSON lists (Week 3 used "a; b; c" strings - see
  ``CareerNLP.to_week3_dict`` for the legacy shape).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .version import ASSET_VERSION, CONFIDENCE_METHOD, SCHEMA_VERSION

ConfidenceLevel = Literal["low", "medium", "high"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
class Confidence(_Model):
    """Confidence attached to a result.

    ``score`` is in [0, 1]. For classification it is the model's predicted
    probability. For matching/recommendation it is an *evidence-sufficiency*
    heuristic (how much signal the result rests on) - it is NOT a calibrated
    probability that the answer is correct. ``calibrated`` says which it is.
    """

    score: float = Field(..., ge=0.0, le=1.0)
    level: ConfidenceLevel
    method: str = CONFIDENCE_METHOD
    calibrated: bool = False
    factors: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-factor breakdown. Only populated when include_details=true.",
    )


class ModelInfo(_Model):
    embedding_backend: str
    embedding_model: str
    jobs_indexed: int


class Meta(_Model):
    asset_version: str = ASSET_VERSION
    schema_version: str = SCHEMA_VERSION
    model: ModelInfo
    processing_ms: float
    warnings: List[str] = Field(default_factory=list)


class _Request(_Model):
    pass


# ---------------------------------------------------------------------------
# Extraction (NER / information extraction)
# ---------------------------------------------------------------------------
class ExtractedSkill(_Model):
    name: str
    mentions: int = Field(..., ge=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ExtractedDegree(_Model):
    text: str = Field(..., description="Normalised, e.g. 'B.Tech in Data Science'.")
    degree: str
    specialization: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class ExtractedContact(_Model):
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class TextRequest(_Request):
    text: str = Field(..., min_length=1, description="CV / profile / job text.")


class ExtractResponse(_Model):
    skills: List[ExtractedSkill]
    degrees_or_certifications: List[ExtractedDegree]
    emails: List[ExtractedContact]
    phone_numbers: List[ExtractedContact]
    experience_mentions: List[str]
    experience_years: Optional[float] = Field(
        default=None, description="Largest explicit 'N years' mention, if any."
    )
    entities_count: int
    meta: Meta


class StudentProfile(_Model):
    normalized_education: str
    skills: List[str]
    experience: str
    profile_summary: str = Field(
        ..., description="The text actually embedded for semantic similarity."
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
class ClassifyRequest(TextRequest):
    top_k: int = Field(default=5, ge=1, le=20)


class CategoryScore(_Model):
    category: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ClassifyResponse(_Model):
    predicted_category: str
    confidence: Confidence
    top_categories: List[CategoryScore]
    meta: Meta


# ---------------------------------------------------------------------------
# Embeddings / similarity / summarisation
# ---------------------------------------------------------------------------
class EmbedRequest(_Request):
    texts: List[str] = Field(..., min_length=1, max_length=64)


class EmbedResponse(_Model):
    model: str
    dimensions: int
    vectors: List[List[float]]
    meta: Meta


class SimilarityRequest(_Request):
    text_a: str = Field(..., min_length=1)
    text_b: str = Field(..., min_length=1)


class SimilarityResponse(_Model):
    similarity: float = Field(..., description="Cosine similarity, -1..1 (typically 0..1).")
    similarity_percent: float = Field(..., description="similarity*100 clipped to 0..100.")
    model: str
    meta: Meta


class SummarizeRequest(TextRequest):
    max_sentences: Optional[int] = Field(default=None, ge=1, le=20)


class SummarySentence(_Model):
    index: int
    text: str
    relevance: float = Field(..., description="Cosine similarity to the document centroid.")


class SummarizeResponse(_Model):
    summary: str
    sentences_total: int
    sentences_used: int
    selected: List[SummarySentence]
    meta: Meta


# ---------------------------------------------------------------------------
# Retrieval ("RAG" retrieval stage)
# ---------------------------------------------------------------------------
class RetrieveRequest(_Request):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedJob(_Model):
    rank: int
    job_id: str
    category: str
    job_title: str
    similarity: float
    skills: List[str]
    job_description: str


class RetrieveResponse(_Model):
    query: str
    results: List[RetrievedJob]
    context: str = Field(..., description="Concatenated job text, ready to feed a generator.")
    meta: Meta


# ---------------------------------------------------------------------------
# Matching - JD provided
# ---------------------------------------------------------------------------
class MatchRequest(_Request):
    student_text: str = Field(..., min_length=1)
    job_text: str = Field(..., min_length=1)
    include_details: bool = Field(
        default=False,
        description="Add score_breakdown, student_profile and confidence factors.",
    )


class ScoreBreakdown(_Model):
    skill_match_percent: float
    semantic_similarity_percent: float
    category_compatibility_percent: float
    skill_weight: float
    semantic_weight: float
    category_weight: float
    required_skills_detected: int


class MatchResponse(_Model):
    match_score: float = Field(..., ge=0.0, le=100.0)
    matched_skills: List[str]
    missing_weak_skills: List[str]
    recommended_learning: List[str]
    confidence: Confidence
    score_breakdown: Optional[ScoreBreakdown] = None
    student_profile: Optional[StudentProfile] = None
    meta: Meta


# ---------------------------------------------------------------------------
# Recommendation - JD not provided
# ---------------------------------------------------------------------------
class RecommendRequest(_Request):
    student_text: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10, description="Best job + (top_k-1) alternatives.")
    include_details: bool = False


class RecommendedJob(_Model):
    rank: int
    job_id: str
    job_title: str = Field(..., description="Cleaned title (levels / req codes removed).")
    category: str
    match_score: float = Field(..., ge=0.0, le=100.0)
    matched_skills: List[str]
    missing_weak_skills: List[str]
    recommended_learning: List[str]
    score_breakdown: Optional[ScoreBreakdown] = None


class RecommendResponse(_Model):
    recommended_job: RecommendedJob
    alternatives: List[RecommendedJob]
    confidence: Confidence
    student_profile: Optional[StudentProfile] = None
    meta: Meta


class AnalyzeRequest(_Request):
    student_text: str = Field(..., min_length=1)
    job_text: Optional[str] = Field(
        default=None, description="If omitted, the recommendation flow is used."
    )
    top_k: int = Field(default=3, ge=1, le=10)
    include_details: bool = False


class AnalyzeResponse(_Model):
    flow: Literal["jd_provided", "jd_not_provided"]
    match: Optional[MatchResponse] = None
    recommendation: Optional[RecommendResponse] = None


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
class CVResponse(_Model):
    filename: str
    characters_extracted: int
    text: str
    entities: ExtractResponse
    profile: StudentProfile
    meta: Meta


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class HealthResponse(_Model):
    status: Literal["ok", "degraded"]
    ready: bool
    asset_version: str
    schema_version: str
    embedding_backend: Optional[str] = None
    jobs_indexed: Optional[int] = None


class InfoResponse(_Model):
    asset_version: str
    schema_version: str
    confidence_method: str
    embedding_backend: str
    embedding_model: str
    jobs_indexed: int
    categories: List[str]
    skill_vocabulary_size: int
    scoring: Dict[str, float]
    limits: Dict[str, int]


class ErrorBody(_Model):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None


class ErrorResponse(_Model):
    error: ErrorBody
