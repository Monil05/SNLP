"""CareerNLP - the reusable engine.

One object, lazily initialised, thread-safe after warm-up, returning validated
Pydantic results. It owns: job dataset + skill vocabulary, category classifier,
embedding backend + job index (cached on disk), and the match/recommend logic
(Match Score = 0.70 x Skill Match % + 0.30 x Semantic Similarity %).
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from . import confidence as conf
from .config import AssetConfig
from .embedders import Embedder, LSAEmbedder, SentenceTransformerEmbedder
from .errors import CVProcessingError, InputValidationError, ModelLoadError
from .extraction import (
    extract_degrees,
    extract_emails,
    extract_experience,
    extract_phones,
    extract_skills,
)
from .schemas import (
    AnalyzeResponse,
    CategoryScore,
    ClassifyResponse,
    CVResponse,
    EmbedResponse,
    ExtractResponse,
    InfoResponse,
    MatchResponse,
    Meta,
    ModelInfo,
    RecommendedJob,
    RecommendResponse,
    RetrievedJob,
    RetrieveResponse,
    ScoreBreakdown,
    SimilarityResponse,
    StudentProfile,
    SummarizeResponse,
    SummarySentence,
)
from .text import (
    SkillExtractor,
    build_job_text,
    clean_job_title,
    clean_text,
    format_skills,
    parse_skills,
)
from .version import ASSET_VERSION, CONFIDENCE_METHOD, SCHEMA_VERSION

log = logging.getLogger("career_nlp")

REQUIRED_JOB_COLUMNS = ["job_id", "category", "job_title", "job_description", "job_skill_set"]


@dataclass
class _State:
    jobs: pd.DataFrame
    job_ids: List[str]
    job_titles: List[str]
    job_categories: List[str]
    job_descriptions: List[str]
    job_skill_sets: List[set]
    skill_extractor: SkillExtractor
    skill_demand: Counter
    classifier_vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    embedder: Embedder
    backend: str
    index_vectors: np.ndarray
    warnings: List[str] = field(default_factory=list)


class CareerNLP:
    """Reusable NLP component for CV/job analysis, matching and recommendation.

    Parameters
    ----------
    config:   ``AssetConfig`` (defaults to the repo's data locations).
    jobs:     optional DataFrame with columns ``job_id, category, job_title,
              job_description, job_skill_set``. Overrides ``config.jobs_path``.
    embedder: optional custom ``Embedder``. Overrides ``config.embedding_backend``.

    Nothing heavy happens in ``__init__``; the first call (or ``warmup()``)
    builds the index.
    """

    def __init__(
        self,
        config: Optional[AssetConfig] = None,
        jobs: Optional[pd.DataFrame] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.config = config or AssetConfig()
        self._jobs_arg = jobs
        self._embedder_arg = embedder
        self._state: Optional[_State] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._state is not None

    def warmup(self) -> "CareerNLP":
        self._ensure()
        return self

    def _ensure(self) -> _State:
        if self._state is not None:
            return self._state
        with self._lock:
            if self._state is None:
                t0 = time.perf_counter()
                self._state = self._build_state()
                log.info("CareerNLP ready in %.1fs (%s, %d jobs)",
                         time.perf_counter() - t0, self._state.backend, len(self._state.jobs))
        return self._state

    def _load_jobs(self) -> pd.DataFrame:
        if self._jobs_arg is not None:
            jobs = self._jobs_arg.copy()
        else:
            path = Path(self.config.jobs_path)
            if not path.exists():
                raise ModelLoadError(
                    f"Job dataset not found at {path}. Set AssetConfig(jobs_path=...) "
                    "or CAREER_NLP_JOBS_PATH.", {"jobs_path": str(path)})
            jobs = pd.read_csv(path)
        missing = [c for c in REQUIRED_JOB_COLUMNS if c not in jobs.columns]
        if missing:
            raise ModelLoadError(
                f"Job dataset is missing required columns: {missing}",
                {"required": REQUIRED_JOB_COLUMNS, "missing": missing})
        if jobs.empty:
            raise ModelLoadError("Job dataset is empty.")
        return jobs.reset_index(drop=True)

    def _make_embedder(self, corpus: Sequence[str], warnings: List[str]) -> Tuple[Embedder, str]:
        if self._embedder_arg is not None:
            return self._embedder_arg, "custom"
        backend = self.config.embedding_backend
        if backend in ("sentence-transformers", "auto"):
            try:
                emb = SentenceTransformerEmbedder(self.config.embedding_model, self.config.model_cache_dir)
                return emb, "sentence-transformers"
            except ModelLoadError as exc:
                if backend == "sentence-transformers":
                    raise
                warnings.append(
                    "Sentence-Transformer unavailable; using the LSA offline fallback. "
                    "Scores are not comparable with the Week 3 core model. "
                    f"Reason: {exc.message}")
        return LSAEmbedder(corpus, self.config.lsa_dimensions), "lsa"

    def _job_index(self, embedder: Embedder, backend: str, ids: List[str], texts: List[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        if backend == "lsa":  # cheap to recompute, and the embedder is fitted per process
            return embedder.encode(cleaned)
        digest = hashlib.sha256()
        digest.update(embedder.name.encode())
        for i, t in zip(ids, cleaned):
            digest.update(i.encode()); digest.update(b"\0"); digest.update(t.encode()); digest.update(b"\1")
        cache = Path(self.config.cache_dir) / f"job_index_{digest.hexdigest()[:20]}.npz"
        if cache.exists():
            try:
                return np.load(cache, allow_pickle=False)["embeddings"]
            except Exception:  # corrupt cache -> rebuild
                log.warning("Ignoring unreadable job-index cache %s", cache)
        vectors = np.asarray(embedder.encode(cleaned), dtype=np.float32)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, embeddings=vectors)
        except OSError:
            log.warning("Could not write job-index cache %s (continuing without it)", cache)
        return vectors

    def _build_state(self) -> _State:
        jobs = self._load_jobs()
        warnings: List[str] = []
        ids = jobs["job_id"].astype(str).tolist()
        job_texts = [build_job_text(r) for r in jobs.to_dict("records")]
        skill_sets = [set(parse_skills(v)) for v in jobs["job_skill_set"]]

        extractor = SkillExtractor.from_jobs(jobs)
        demand: Counter = Counter()
        for s in skill_sets:
            demand.update(s)

        # Category classifier. Trained on title + description only: the Week 3
        # classifier also saw the category name inside its own training text.
        clf_texts = [
            f"{clean_text(t).lower()} {clean_text(d).lower()}"
            for t, d in zip(jobs["job_title"], jobs["job_description"])
        ]
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vec.fit_transform(clf_texts)
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(matrix, jobs["category"].astype(str))

        embedder, backend = self._make_embedder(job_texts, warnings)
        index = self._job_index(embedder, backend, ids, job_texts)

        return _State(
            jobs=jobs, job_ids=ids,
            job_titles=jobs["job_title"].astype(str).tolist(),
            job_categories=jobs["category"].astype(str).tolist(),
            job_descriptions=jobs["job_description"].astype(str).tolist(),
            job_skill_sets=skill_sets, skill_extractor=extractor, skill_demand=demand,
            classifier_vectorizer=vec, classifier=classifier,
            embedder=embedder, backend=backend, index_vectors=index, warnings=warnings,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _check_text(self, value: Any, name: str) -> str:
        if not isinstance(value, str):
            raise InputValidationError(f"'{name}' must be a string.", {"field": name})
        if not value.strip():
            raise InputValidationError(f"'{name}' must not be empty.", {"field": name})
        if len(value) > self.config.max_text_chars:
            raise InputValidationError(
                f"'{name}' is too long ({len(value)} characters; max {self.config.max_text_chars}).",
                {"field": name, "length": len(value), "max": self.config.max_text_chars})
        return value

    def _top_k(self, k: Optional[int], default: int, maximum: int) -> int:
        k = default if k is None else k
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise InputValidationError("'top_k' must be a positive integer.", {"field": "top_k"})
        return min(k, maximum)

    def _meta(self, st: _State, t0: float, warnings: Optional[List[str]] = None) -> Meta:
        return Meta(
            model=ModelInfo(embedding_backend=st.backend, embedding_model=st.embedder.name,
                            jobs_indexed=len(st.job_ids)),
            processing_ms=round((time.perf_counter() - t0) * 1000, 1),
            warnings=list(st.warnings) + list(warnings or []),
        )

    def _embed(self, st: _State, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(st.embedder.encode([clean_text(t) for t in texts]), dtype=np.float32)

    @staticmethod
    def _cos_percent(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.clip(cosine_similarity(a, b).ravel() * 100.0, 0.0, 100.0)

    def _order_learning(self, st: _State, missing: Sequence[str]) -> List[str]:
        """Most in-demand (across the job dataset) first, then alphabetical."""
        return sorted(missing, key=lambda s: (-st.skill_demand.get(s, 0), s))

    def _combine(self, skill_pct: float, sem_pct: float) -> float:
        return self.config.skill_weight * skill_pct + self.config.semantic_weight * sem_pct

    def _breakdown(self, skill_pct: float, sem_pct: float, n_required: int) -> ScoreBreakdown:
        return ScoreBreakdown(
            skill_match_percent=round(skill_pct, 2),
            semantic_similarity_percent=round(sem_pct, 2),
            skill_weight=self.config.skill_weight,
            semantic_weight=self.config.semantic_weight,
            required_skills_detected=n_required,
        )

    # ------------------------------------------------------------------
    # information extraction
    # ------------------------------------------------------------------
    def _extract_fields(self, st: _State, text: str) -> Dict[str, Any]:
        text = clean_text(text)
        skills = extract_skills(text, st.skill_extractor)
        degrees = extract_degrees(text)
        emails = extract_emails(text)
        phones = extract_phones(text)
        mentions, years = extract_experience(text)
        return dict(
            skills=skills, degrees_or_certifications=degrees, emails=emails,
            phone_numbers=phones, experience_mentions=mentions, experience_years=years,
            entities_count=len(skills) + len(degrees) + len(emails) + len(phones) + len(mentions),
        )

    def extract(self, text: str) -> ExtractResponse:
        t0 = time.perf_counter()
        self._check_text(text, "text")
        st = self._ensure()
        return ExtractResponse(**self._extract_fields(st, text), meta=self._meta(st, t0))

    def _profile_from_fields(self, fields: Dict[str, Any]) -> StudentProfile:
        education = "; ".join(d.text for d in fields["degrees_or_certifications"]) or "Not specified"
        skills = [s.name for s in fields["skills"]]
        skills_text = "; ".join(skills) if skills else "Not specified"
        experience = "; ".join(fields["experience_mentions"]) or "Not specified"
        return StudentProfile(
            normalized_education=education, skills=skills, experience=experience,
            profile_summary=f"Education: {education}. Skills: {skills_text}. Experience: {experience}.",
        )

    def derive_student_profile(self, text: str) -> StudentProfile:
        self._check_text(text, "text")
        st = self._ensure()
        return self._profile_from_fields(self._extract_fields(st, text))

    # ------------------------------------------------------------------
    # classification
    # ------------------------------------------------------------------
    def classify(self, text: str, top_k: int = 5) -> ClassifyResponse:
        t0 = time.perf_counter()
        self._check_text(text, "text")
        k = self._top_k(top_k, 5, 20)
        st = self._ensure()
        proba = st.classifier.predict_proba(st.classifier_vectorizer.transform([clean_text(text)]))[0]
        order = np.argsort(proba)[::-1][:k]
        top = [CategoryScore(category=str(st.classifier.classes_[i]), probability=round(float(proba[i]), 4))
               for i in order]
        runner_up = top[1].probability if len(top) > 1 else None
        confidence = conf.classification_confidence(top[0].probability, runner_up)
        warnings = []
        if confidence.level == "low":
            warnings.append("Low classification confidence: the text may be too short or span several categories.")
        return ClassifyResponse(predicted_category=top[0].category, confidence=confidence,
                                top_categories=top, meta=self._meta(st, t0, warnings))

    # ------------------------------------------------------------------
    # embeddings / similarity / summarisation
    # ------------------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> EmbedResponse:
        t0 = time.perf_counter()
        if not isinstance(texts, (list, tuple)) or not texts:
            raise InputValidationError("'texts' must be a non-empty list of strings.", {"field": "texts"})
        for i, t in enumerate(texts):
            self._check_text(t, f"texts[{i}]")
        st = self._ensure()
        vectors = self._embed(st, texts)
        return EmbedResponse(model=st.embedder.name, dimensions=int(vectors.shape[1]),
                             vectors=vectors.tolist(), meta=self._meta(st, t0))

    def similarity(self, text_a: str, text_b: str) -> SimilarityResponse:
        t0 = time.perf_counter()
        self._check_text(text_a, "text_a"); self._check_text(text_b, "text_b")
        st = self._ensure()
        v = self._embed(st, [text_a, text_b])
        score = float(cosine_similarity(v[0:1], v[1:2])[0][0])
        return SimilarityResponse(similarity=round(score, 4),
                                  similarity_percent=round(float(np.clip(score * 100, 0, 100)), 2),
                                  model=st.embedder.name, meta=self._meta(st, t0))

    def summarize(self, text: str, max_sentences: Optional[int] = None) -> SummarizeResponse:
        t0 = time.perf_counter()
        self._check_text(text, "text")
        st = self._ensure()
        n = self._top_k(max_sentences, self.config.summary_sentences, 20)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text(text)) if s.strip()]
        if len(sentences) <= n:
            chosen, scores = list(range(len(sentences))), [1.0] * len(sentences)
        else:
            matrix = st.classifier_vectorizer.transform(sentences)
            centroid = np.asarray(matrix.mean(axis=0))
            scores = cosine_similarity(matrix, centroid).ravel().tolist()
            chosen = sorted(np.argsort(scores)[-n:].tolist())
        selected = [SummarySentence(index=i, text=sentences[i], relevance=round(float(scores[i]), 4)) for i in chosen]
        return SummarizeResponse(summary=" ".join(s.text for s in selected), sentences_total=len(sentences),
                                 sentences_used=len(selected), selected=selected, meta=self._meta(st, t0))

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> RetrieveResponse:
        t0 = time.perf_counter()
        self._check_text(query, "query")
        st = self._ensure()
        k = self._top_k(top_k, self.config.default_top_k, min(self.config.max_top_k, len(st.job_ids)))
        scores = cosine_similarity(self._embed(st, [query]), st.index_vectors).ravel()
        idx = np.argsort(-scores, kind="stable")[:k]
        results = [
            RetrievedJob(rank=r + 1, job_id=st.job_ids[i], category=st.job_categories[i],
                         job_title=st.job_titles[i], similarity=round(float(scores[i]), 4),
                         skills=sorted(st.job_skill_sets[i]), job_description=st.job_descriptions[i])
            for r, i in enumerate(map(int, idx))
        ]
        context = "\n\n".join(f"Job {r.job_id} | {r.job_title} | {r.category}\n{r.job_description}" for r in results)
        return RetrieveResponse(query=query, results=results, context=context, meta=self._meta(st, t0))

    # ------------------------------------------------------------------
    # matching - JD provided
    # ------------------------------------------------------------------
    def match(self, student_text: str, job_text: str, include_details: bool = False) -> MatchResponse:
        t0 = time.perf_counter()
        self._check_text(student_text, "student_text"); self._check_text(job_text, "job_text")
        st = self._ensure()

        fields = self._extract_fields(st, student_text)
        profile = self._profile_from_fields(fields)
        student_skills = set(profile.skills)
        job_skills = set(st.skill_extractor.extract(job_text))

        matched = sorted(student_skills & job_skills)
        missing = sorted(job_skills - student_skills)
        skill_pct = (len(matched) / len(job_skills) * 100.0) if job_skills else 0.0

        sv, jv = self._embed(st, [profile.profile_summary]), self._embed(st, [job_text])
        sem_pct = float(self._cos_percent(sv, jv)[0])
        score = round(self._combine(skill_pct, sem_pct), 2)

        warnings: List[str] = []
        if not job_skills:
            warnings.append("No known skills were recognised in the job description, so Skill Match is 0 and "
                            f"the score reflects semantic similarity only (max {self.config.semantic_weight * 100:.0f}).")
        if not student_skills:
            warnings.append("No known skills were recognised in the student text; Skill Match is 0.")

        confidence = conf.match_confidence(len(job_skills), len(student_skills), include_factors=include_details)
        return MatchResponse(
            match_score=score, matched_skills=matched, missing_weak_skills=missing,
            recommended_learning=self._order_learning(st, missing), confidence=confidence,
            score_breakdown=self._breakdown(skill_pct, sem_pct, len(job_skills)) if include_details else None,
            student_profile=profile if include_details else None,
            meta=self._meta(st, t0, warnings),
        )

    # ------------------------------------------------------------------
    # recommendation - JD not provided
    # ------------------------------------------------------------------
    def _job_record(self, st: _State, i: int, rank: int, student_skills: set,
                    sem_pct: float, details: bool) -> Tuple[RecommendedJob, float]:
        required = st.job_skill_sets[i]
        matched = sorted(student_skills & required)
        missing = sorted(required - student_skills)
        skill_pct = (len(matched) / len(required) * 100.0) if required else 0.0
        score = round(self._combine(skill_pct, sem_pct), 2)
        rec = RecommendedJob(
            rank=rank, job_id=st.job_ids[i], job_title=clean_job_title(st.job_titles[i]),
            category=st.job_categories[i], match_score=score, matched_skills=matched,
            missing_weak_skills=missing, recommended_learning=self._order_learning(st, missing),
            score_breakdown=self._breakdown(skill_pct, sem_pct, len(required)) if details else None,
        )
        return rec, skill_pct

    def recommend(self, student_text: str, top_k: int = 3, include_details: bool = False) -> RecommendResponse:
        t0 = time.perf_counter()
        self._check_text(student_text, "student_text")
        k = self._top_k(top_k, 3, 10)
        st = self._ensure()

        profile = self._profile_from_fields(self._extract_fields(st, student_text))
        student_skills = set(profile.skills)
        sem = self._cos_percent(self._embed(st, [profile.profile_summary]), st.index_vectors)

        n_matched = np.array([len(student_skills & r) for r in st.job_skill_sets], dtype=np.float64)
        n_req = np.array([len(r) for r in st.job_skill_sets], dtype=np.float64)
        skill = np.divide(n_matched, n_req, out=np.zeros_like(n_matched), where=n_req > 0) * 100.0
        scores = np.round(self.config.skill_weight * skill + self.config.semantic_weight * sem, 2)
        order = np.argsort(-scores, kind="stable")

        # alternatives: next-best jobs whose cleaned title differs from those already shown
        picked: List[int] = []
        seen_titles = set()
        for i in map(int, order):
            title = clean_job_title(st.job_titles[i]).lower()
            if title in seen_titles:
                continue
            seen_titles.add(title); picked.append(i)
            if len(picked) == k:
                break

        records, top_skill_pct = [], 0.0
        for rank, i in enumerate(picked, start=1):
            rec, skill_pct = self._job_record(st, i, rank, student_skills, float(sem[i]), include_details)
            records.append(rec)
            if rank == 1:
                top_skill_pct = skill_pct

        top5 = [st.job_categories[int(i)] for i in order[:5]]
        agreement = top5.count(st.job_categories[int(order[0])]) / len(top5)
        second = float(scores[int(order[1])]) if len(order) > 1 else None
        confidence = conf.recommend_confidence(
            top_score=float(scores[int(order[0])]), second_score=second,
            top_skill_match_percent=top_skill_pct, category_agreement=agreement,
            n_student_skills=len(student_skills), include_factors=include_details)

        warnings: List[str] = []
        if not student_skills:
            warnings.append("No known skills were recognised in the student text; the recommendation relies on "
                            "semantic similarity alone and is unreliable.")
        if confidence.level == "low" and student_skills:
            warnings.append("Low confidence: top results are close in score and/or span several job categories.")

        return RecommendResponse(
            recommended_job=records[0], alternatives=records[1:], confidence=confidence,
            student_profile=profile if include_details else None, meta=self._meta(st, t0, warnings))

    def analyze(self, student_text: str, job_text: Optional[str] = None, top_k: int = 3,
                include_details: bool = False) -> AnalyzeResponse:
        """Single entry point for both flows: JD given -> match, else -> recommend."""
        if job_text is not None and job_text.strip():
            return AnalyzeResponse(flow="jd_provided", match=self.match(student_text, job_text, include_details))
        return AnalyzeResponse(flow="jd_not_provided",
                               recommendation=self.recommend(student_text, top_k, include_details))

    # ------------------------------------------------------------------
    # CV processing
    # ------------------------------------------------------------------
    def _read_cv_text(self, data: bytes, filename: str) -> str:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in {".pdf", ".docx"}:
            raise CVProcessingError("Only .pdf and .docx CVs are supported.", {"filename": filename, "suffix": suffix})
        if len(data) > self.config.max_cv_bytes:
            raise CVProcessingError(f"CV is too large ({len(data)} bytes; max {self.config.max_cv_bytes}).",
                                    {"max_bytes": self.config.max_cv_bytes})
        try:
            if suffix == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                return "\n".join((p.extract_text() or "") for p in reader.pages)
            from docx import Document
            doc = Document(io.BytesIO(data))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:  # many CVs keep skills/education in tables
                for row in table.rows:
                    parts.extend(cell.text for cell in row.cells)
            return "\n".join(parts)
        except ImportError as exc:
            raise CVProcessingError(f"Missing dependency for {suffix} files: {exc.name}. "
                                    "Install with: pip install pypdf python-docx", {"package": exc.name}) from exc
        except Exception as exc:
            raise CVProcessingError(f"Could not read the {suffix} file: {exc}", {"filename": filename}) from exc

    def analyze_cv_bytes(self, data: bytes, filename: str) -> CVResponse:
        t0 = time.perf_counter()
        text = self._read_cv_text(data, filename)
        if not text.strip():
            raise CVProcessingError(
                "No text could be extracted (scanned/image-only CVs need OCR, which is not included).",
                {"filename": filename})
        st = self._ensure()
        text = text[: self.config.max_text_chars]
        fields = self._extract_fields(st, text)
        warnings = []
        if len(text.strip()) < 200:
            warnings.append("Very little text was extracted; results may be incomplete.")
        meta = self._meta(st, t0, warnings)
        return CVResponse(filename=Path(filename).name, characters_extracted=len(text), text=text,
                          entities=ExtractResponse(**fields, meta=meta),
                          profile=self._profile_from_fields(fields), meta=meta)

    def analyze_cv(self, path: Union[str, Path]) -> CVResponse:
        path = Path(path)
        if not path.exists():
            raise CVProcessingError(f"File not found: {path}", {"path": str(path)})
        return self.analyze_cv_bytes(path.read_bytes(), path.name)

    # ------------------------------------------------------------------
    # service info / compatibility
    # ------------------------------------------------------------------
    def info(self) -> InfoResponse:
        st = self._ensure()
        return InfoResponse(
            asset_version=ASSET_VERSION, schema_version=SCHEMA_VERSION, confidence_method=CONFIDENCE_METHOD,
            embedding_backend=st.backend, embedding_model=st.embedder.name, jobs_indexed=len(st.job_ids),
            categories=sorted(set(st.job_categories)), skill_vocabulary_size=len(st.skill_extractor),
            scoring={"skill_weight": self.config.skill_weight, "semantic_weight": self.config.semantic_weight},
            limits={"max_text_chars": self.config.max_text_chars, "max_cv_bytes": self.config.max_cv_bytes,
                    "max_top_k": self.config.max_top_k},
        )

    @staticmethod
    def to_week3_dict(result: Union[MatchResponse, RecommendResponse]) -> Dict[str, Any]:
        """Convert a result to the exact Week 3 output shape ('a; b; c' strings,
        no confidence, no job_id) for callers that still expect it."""
        if isinstance(result, MatchResponse):
            return {"match_score": result.match_score,
                    "matched_skills": format_skills(result.matched_skills),
                    "missing_weak_skills": format_skills(result.missing_weak_skills),
                    "recommended_learning": format_skills(result.recommended_learning)}
        job = result.recommended_job
        return {"recommended_job": job.job_title, "match_score": job.match_score,
                "matched_skills": format_skills(job.matched_skills),
                "missing_weak_skills": format_skills(job.missing_weak_skills),
                "recommended_learning": format_skills(job.recommended_learning)}
