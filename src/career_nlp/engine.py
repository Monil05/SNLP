"""CareerNLP - the reusable engine.

One object, lazily initialised, thread-safe after warm-up, returning validated
Pydantic results. It owns: job dataset + skill vocabulary, category classifier,
embedding backend + job index (cached on disk), and the match/recommend logic.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
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
    extract_academic_records,
    extract_degrees,
    extract_emails,
    extract_experience,
    extract_phones,
    extract_skills,
)
from .schemas import (
    AcademicAdjustment,
    AcademicRecord,
    AnalyzeResponse,
    CategoryScore,
    CompetencyRequirement,
    ComparedJob,
    CompareJobsResponse,
    ClassifyResponse,
    CVResponse,
    EmbedResponse,
    ExtractResponse,
    InfoResponse,
    MatchResponse,
    Meta,
    JobDescriptionInput,
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
    normalize_skill,
    parse_skills,
)
from .version import ASSET_VERSION, CONFIDENCE_METHOD, SCHEMA_VERSION

log = logging.getLogger("career_nlp")

REQUIRED_JOB_COLUMNS = ["job_id", "category", "job_title", "job_description", "job_skill_set"]
ACADEMIC_REQUIREMENT_MIN_SIMILARITY = 0.43
ACADEMIC_REQUIREMENT_MIN_MARGIN = 0.025
ACADEMIC_DOMAIN_MIN_SIMILARITY = 0.36
ACADEMIC_DOMAIN_MIN_MARGIN = 0.02
ACADEMIC_DOMAIN_MARGIN_RANGE = 0.18
COMPETENCY_SEMANTIC_MATCH_THRESHOLD = 0.62
COMPETENCY_LOGIC_VERSION = "6"
NEAR_DUPLICATE_COSINE = 0.97
NEAR_DUPLICATE_SKILL_JACCARD = 0.90
SECTION_WINDOW = 70

# Canonical skill plus paraphrases. Used for matching and for collapsing
# wording variants into one competency. Heads are the stored vocabulary form.
_ALIAS_GROUPS = (
    ("sql", (
        "sql", "structured query language", "relational database", "relational databases",
        "relational database querying", "relational database concepts", "sql querying",
        "sql queries", "writing queries",
    )),
    ("python", (
        "python", "python programming", "python development", "python scripting", "python script",
    )),
    ("etl", (
        "etl", "extract transform load", "data pipeline", "data pipelines",
        "data pipeline development", "data ingestion",
    )),
    ("data warehousing", (
        "data warehousing", "data warehouse", "data warehouses", "warehouse modelling",
        "warehouse modeling", "data warehouse modelling", "data warehouse modeling",
    )),
    ("data visualization", (
        "data visualization", "data visualisation", "data visualisations", "data visualizations",
    )),
    ("data analysis", (
        "data analysis", "data analytics",
    )),
    ("machine learning", (
        "machine learning",
    )),
    ("nlp", (
        "nlp", "natural language processing",
    )),
    ("power bi", (
        "power bi", "powerbi",
    )),
)

_SECTION_PATTERNS = (
    ("preferred", re.compile(
        r"skills that will help you stand out|preferred qualifications|preferred skills|"
        r"nice[- ]to[- ]have|good to have|desired qualifications|bonus qualifications|"
        r"qualifications that (?:will|would) set you apart|stand[- ]out skills",
        re.I,
    )),
    ("required", re.compile(
        r"\b(?:minimum|basic)\s+qualifications\b|\brequired qualifications\b|\brequired skills\b|"
        r"\bwho you are\b|\bwhat you(?:'|’)ll need\b|\bmust[- ]have skills\b",
        re.I,
    )),
    ("responsibilities", re.compile(
        r"\bwhat you(?:'|’)ll do\b|\bresponsibilities\b|\bjob purpose\b",
        re.I,
    )),
    ("other", re.compile(
        r"\bsalary\b|\bbenefits\b|\babout (?:us|the company)\b|\badditional information\b|"
        r"\bequal opportunity\b|\bour culture\b|\bhow to apply\b|\bwe offer\b",
        re.I,
    )),
)
_HARD_REQUIRED_RE = re.compile(
    r"\b(?:must(?:\s+have)?|requires|required|require|mandatory|essential|necessary)\b",
    re.I,
)
_EXPERIENCE_REQUIRED_RE = re.compile(
    r"\b(?:proficient|skilled)\s+in\b|\bexperience\s+(?:with|in|writing|using|developing)\b|"
    r"\bexperienced\s+in\b|"
    r"\byou\s+have\s+experience\b|\bexpertise\s+in\b|\bknowledge\s+of\b|"
    r"\bstrong\s+(?:knowledge|expertise|skills?)\s+(?:of|in|with)\b|"
    r"\bproficient\s+understanding\s+of\b",
    re.I,
)
# "Required to work/collaborate" describes a duty, not a competency the candidate must already have.
_DUTY_REQUIRED_RE = re.compile(
    r"\brequired\s+to\s+(?:work|collaborate|partner|assist|support|perform|complete|"
    r"attend|travel|participate|contribute|report|operate|maintain|provide|deliver|"
    r"coordinate|liaise|interface)\b",
    re.I,
)
_CUE_PROXIMITY = 90
# Last tokens that are too broad to stand in for a multi-word competency.
_VENDOR_PREFIXES = frozenset({
    "microsoft", "ms", "google", "amazon", "aws", "apache", "ibm", "oracle", "adobe",
})
_TRAILING_TOKEN_STOP = frozenset({
    "ability", "abilities", "administration", "analysis", "applications", "concepts",
    "design", "development", "engineering", "experience", "fundamentals", "knowledge",
    "learning", "management", "methods", "modelling", "modeling", "operations",
    "planning", "principles", "processing", "reporting", "research", "services",
    "service", "skill", "skills", "software", "support", "system", "systems",
    "technology", "testing", "tools", "training", "communication", "collaboration",
    "leadership", "teamwork",
})
_METADATA_EVIDENCE = (
    "Listed in the job skill metadata without an explicit requirement statement in the description."
)
_STUDENT_NEGATIVE_RE = re.compile(
    r"\b(?:no|without|lacking|zero)\b(?:\s+\w+){0,3}\s*$|"
    r"\b(?:no|without|lack(?:ing)?(?:\s+of)?|zero)\s+(?:any\s+|prior\s+|previous\s+)?"
    r"(?:experience|knowledge|skills?|background|familiarity|proficiency)\s+(?:with|in|of)\s*$|"
    r"\b(?:never\s+(?:used|worked\s+with)|do\s+not\s+know|don't\s+know|not\s+familiar\s+with|"
    r"not\s+(?:yet\s+)?(?:experienced|proficient|skilled)\s+(?:with|in))\s*$",
    re.I,
)
_STUDENT_ASPIRATIONAL_RE = re.compile(
    r"\b(?:interested\s+in|curious\s+about|want(?:s)?\s+to\s+learn|"
    r"hope(?:s)?\s+to\s+learn|plan(?:s)?\s+to\s+learn|aim(?:s)?\s+to\s+learn|"
    r"would\s+like\s+to\s+learn|looking\s+to\s+learn|aspire(?:s)?\s+to|"
    r"wish(?:es)?\s+to\s+learn)\b",
    re.I,
)
_STUDENT_DEMONSTRATED_RE = re.compile(
    r"\b(?:skills?|tools?|technologies|competenc(?:y|ies)|proficien\w*|experienced|"
    r"experience|knowledge|expertise|understanding|familiarity|familiar|worked|used|built|"
    r"implemented|developed|applied|created|evaluated|deployed|analysed|analyzed|designed|"
    r"maintained|automated|trained|studied|coursework|certification|certified|internship|"
    r"project|projects|hands[- ]on|practical|demonstrated|programmed|learned|learnt|mastery)\b",
    re.I,
)
_STUDENT_CONTRAST_RE = re.compile(r"\b(?:but|however|although|whereas|yet)\b", re.I)
_STUDENT_LIST_RE = re.compile(r",|;|\band\b|/|\|")
_ALIAS_LIST_CACHE: Dict[str, Tuple[str, ...]] = {}


def _cached_competency_aliases(name: str) -> Tuple[str, ...]:
    found = _ALIAS_LIST_CACHE.get(name)
    if found is None:
        found = tuple(CareerNLP._competency_aliases(name))
        _ALIAS_LIST_CACHE[name] = found
    return found


@lru_cache(maxsize=16384)
def _soft_norm(value: str) -> str:
    text = normalize_skill(value).replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", text).strip()


_ALIAS_LOOKUP: Dict[str, str] = {}
_ALIAS_PHRASES: Dict[str, Tuple[str, ...]] = {}
for _head, _phrases in _ALIAS_GROUPS:
    _canonical = _soft_norm(_head)
    _normalised_phrases = tuple(dict.fromkeys(_soft_norm(phrase) for phrase in _phrases if _soft_norm(phrase)))
    _ALIAS_PHRASES[_canonical] = _normalised_phrases
    for _phrase in _normalised_phrases:
        _ALIAS_LOOKUP[_phrase] = _canonical


@dataclass(frozen=True)
class _Competency:
    name: str
    kind: str
    priority: str
    evidence: str
    generic: bool = False


@dataclass
class _CompetencyAssessment:
    requirements: List[CompetencyRequirement]
    matched: List[str]
    missing: List[str]
    learning: List[str]
    skill_match_percent: float
    matched_aliases: set
    evaluated_count: int
    learning_tiers: Dict[str, int] = field(default_factory=dict)


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
    job_competencies: List[List[_Competency]] = field(default_factory=list)
    job_canonical_skills: List[set] = field(default_factory=list)
    canonical_postings: Dict[str, np.ndarray] = field(default_factory=dict)
    nongeneric_postings: Dict[str, np.ndarray] = field(default_factory=dict)
    # Canonical competency names split the same way _skill_match_percent does,
    # so retrieval can rank jobs by that score without assessing every posting.
    required_terms: List[frozenset] = field(default_factory=list)
    preferred_terms: List[frozenset] = field(default_factory=list)
    generic_terms: List[frozenset] = field(default_factory=list)
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
        self._academic_template_lock = threading.Lock()
        self._academic_template_cache: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._alias_patterns: Dict[str, re.Pattern] = {}
        self._generic_keys = frozenset(_soft_norm(item) for item in self.config.generic_competencies)
        self.last_recommendation_stats: Dict[str, Any] = {}

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

        state = _State(
            jobs=jobs, job_ids=ids,
            job_titles=jobs["job_title"].astype(str).tolist(),
            job_categories=jobs["category"].astype(str).tolist(),
            job_descriptions=jobs["job_description"].astype(str).tolist(),
            job_skill_sets=skill_sets, skill_extractor=extractor, skill_demand=demand,
            classifier_vectorizer=vec, classifier=classifier,
            embedder=embedder, backend=backend, index_vectors=index, warnings=warnings,
        )
        state.job_competencies = self._load_or_extract_competencies(state)
        self._index_job_skills(state)
        return state

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

    def _order_learning(
        self,
        st: _State,
        missing: Sequence[str],
        tiers: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        """Required gaps first, then preferred, then generic; demand breaks ties.

        ``tiers`` maps a normalised skill to 0 (required), 1 (preferred) or
        2 (generic). Missing tiers share one band so older callers keep
        demand order.
        """
        bands = tiers or {}
        return sorted(missing, key=lambda s: (bands.get(s, 1), -st.skill_demand.get(s, 0), s))

    def _combine(self, skill_pct: float, sem_pct: float, category_pct: float) -> float:
        return (
            self.config.skill_weight * skill_pct
            + self.config.semantic_weight * sem_pct
            + self.config.category_weight * category_pct
        )

    def _breakdown(
        self,
        skill_pct: float,
        sem_pct: float,
        category_pct: float,
        n_required: int,
        base_score: float,
        academic_penalty: float = 0.0,
    ) -> ScoreBreakdown:
        return ScoreBreakdown(
            skill_match_percent=round(skill_pct, 2),
            semantic_similarity_percent=round(sem_pct, 2),
            category_compatibility_percent=round(category_pct, 2),
            skill_weight=self.config.skill_weight,
            semantic_weight=self.config.semantic_weight,
            category_weight=self.config.category_weight,
            required_skills_detected=n_required,
            base_match_score=round(base_score, 2),
            academic_penalty_percent=round(academic_penalty, 2),
        )

    @staticmethod
    def _requirement_sentences(text: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?;])\s+|[\r\n]+", text)
            if sentence.strip()
        ] or ([text.strip()] if text.strip() else [])

    @staticmethod
    def _split_requirement_terms(value: str) -> List[str]:
        """Split a competency list without needing a subject/technology dictionary."""
        value = re.split(
            r"\b(?:is|are|was|were|would|will|can|could|may|might|should|must|"
            r"required|required for|mandatory|essential|necessary|preferred|preferable|"
            r"desirable|helpful|useful|beneficial|advantageous|optional|nice[- ]to[- ]have|"
            r"not required|not needed|not mandatory|not essential|not necessary)\b",
            value,
            maxsplit=1,
            flags=re.I,
        )[0]
        value = re.split(
            r"\b(?:to|so that|because|while|when|in order to|as part of)\b",
            value,
            maxsplit=1,
            flags=re.I,
        )[0]
        parts = re.split(r",|\band\b|\bor\b|/|&", value, flags=re.I)
        terms: List[str] = []
        for part in parts:
            term = part.strip(" \t\n\r.:;()[]{}-'\"")
            term = re.sub(
                r"^(?:(?:the|a|an|strong|advanced|deep|expert[- ]level|excellent|good|"
                r"solid|basic|working|practical|hands[- ]on|extensive|comprehensive|"
                r"thorough|relevant|prior|some|basic familiarity with)\s+)+",
                "",
                term,
                flags=re.I,
            ).strip(" \t\n\r.:;()[]{}-'\"")
            term = re.sub(r"\s+", " ", term)
            term = re.sub(
                r"\s+(?:concepts?|methods?|principles?|techniques?|fundamentals?|topics?|areas?|systems?)$",
                "",
                term,
                flags=re.I,
            )
            normalized = normalize_skill(term)
            if (
                len(normalized) < 2
                or len(normalized) > 80
                or normalized in {
                    "it", "this", "that", "these", "those", "the role", "the job",
                    "the position", "the candidate", "the team", "knowledge", "skills",
                    "experience", "expertise", "understanding", "proficiency", "competence",
                }
            ):
                continue
            if normalized not in {normalize_skill(item) for item in terms}:
                terms.append(term)
        return terms

    @staticmethod
    def _requirement_priority(name: str, evidence: str) -> str:
        """Classify a competency as required, preferred, or explicitly excluded."""
        alias = re.escape(normalize_skill(name))
        boundaries = rf"(?<!\w){alias}(?!\w)"
        clauses = [
            clause.strip()
            for clause in re.split(
                r",|;|\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b|\byet\b",
                evidence,
                flags=re.I,
            )
            if clause.strip()
        ] or [evidence]
        relevant_indices = [
            index for index, clause in enumerate(clauses)
            if re.search(boundaries, clause, re.I)
        ]
        if not relevant_indices:
            relevant_indices = [0]

        not_required = re.compile(
            r"\b(?:not\s+(?:required|needed|mandatory|essential|necessary|a\s+prerequisite|"
            r"a\s+requirement)|isn't\s+(?:required|needed|mandatory|essential)|"
            r"doesn't\s+require|do\s+not\s+require|without\s+(?:prior\s+)?"
            r"(?:knowledge|expertise|proficiency|experience|familiarity)|"
            r"no\s+(?:prior\s+)?(?:knowledge|expertise|proficiency|experience|familiarity))\b",
            re.I,
        )
        preferred = re.compile(
            r"\b(?:nice[- ]to[- ]have|good\s+to\s+have|optional|a\s+plus|bonus|"
            r"desirable|preferable|preferred|not\s+mandatory|not\s+essential|"
            r"not\s+necessary|helpful|useful|beneficial|advantageous|can\s+be\s+learned|"
            r"would\s+be\s+(?:helpful|useful|beneficial)|is\s+an\s+advantage)\b",
            re.I,
        )
        standalone_preference = re.compile(
            r"^\s*(?:(?:it|this|that)\s+)?(?:is|would\s+be|remains)?\s*"
            r"(?:still\s+)?(?:nice[- ]to[- ]have|good\s+to\s+have|a\s+plus|bonus|"
            r"desirable|preferable|preferred|helpful|useful|beneficial|advantageous)\s*[.!]?\s*$",
            re.I,
        )

        priorities = []
        for index in relevant_indices:
            clause = clauses[index]
            # When optional wording follows "not required", classify it as preferred.
            next_clause_is_preference = (
                index + 1 < len(clauses)
                and standalone_preference.fullmatch(clauses[index + 1]) is not None
            )
            if preferred.search(clause) or (not_required.search(clause) and next_clause_is_preference):
                priorities.append("preferred")
            elif not_required.search(clause):
                priorities.append("not_required")
            else:
                priorities.append("required")

        if "required" in priorities:
            return "required"
        if "preferred" in priorities:
            return "preferred"
        return "not_required"

    @staticmethod
    @staticmethod
    def _canonical_skill(name: str) -> str:
        key = _soft_norm(name)
        mapped = _ALIAS_LOOKUP.get(key)
        if mapped:
            return mapped
        tokens = key.split()
        if len(tokens) >= 2 and tokens[0] in _VENDOR_PREFIXES:
            tail = tokens[-1]
            if len(tail) >= 4 and tail not in _TRAILING_TOKEN_STOP:
                return _ALIAS_LOOKUP.get(tail, tail)
        return key

    def _alias_pattern(self, alias: str) -> "re.Pattern[str]":
        pattern = self._alias_patterns.get(alias)
        if pattern is None:
            pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.I)
            self._alias_patterns[alias] = pattern
        return pattern

    @staticmethod
    def _competency_aliases(name: str) -> List[str]:
        """Return phrase/acronym aliases inferred from the phrase and known paraphrases."""
        aliases = {normalize_skill(name), _soft_norm(name)}
        without_parentheses = re.sub(r"\s*\([^)]*\)", "", name).strip()
        if without_parentheses:
            aliases.add(normalize_skill(without_parentheses))
            aliases.add(_soft_norm(without_parentheses))
        for acronym in re.findall(r"\(([A-Z][A-Z0-9+#.-]{1,9})\)", name):
            aliases.add(normalize_skill(acronym))
        tokens = _soft_norm(name).split()
        # Only a vendor prefix is safe to drop: "microsoft excel" is "excel".
        # A trailing word such as "datasets" is too broad and creates false matches.
        if len(tokens) >= 2 and tokens[0] in _VENDOR_PREFIXES:
            tail = tokens[-1]
            if len(tail) >= 4 and tail not in _TRAILING_TOKEN_STOP:
                aliases.add(tail)
        canonical = _ALIAS_LOOKUP.get(_soft_norm(name))
        if canonical:
            aliases.update(_ALIAS_PHRASES.get(canonical, ()))
        return sorted((alias for alias in aliases if alias and len(alias) >= 2), key=len, reverse=True)

    def _is_generic_name(self, name: str) -> bool:
        key = _soft_norm(name)
        return key in self._generic_keys or _ALIAS_LOOKUP.get(key, key) in self._generic_keys

    @staticmethod
    def _incidental_skill_tail(text: str, end: int) -> bool:
        """Ignore a skill token that only names a cycle, such as 'reporting cycle'."""
        return re.match(r"\s*cycles?(?![a-z])", text[end:end + 16], re.I) is not None or (
            re.match(r"\s*cycle", text[end:end + 12], re.I) is not None
        )

    def _has_substantive_mention(self, name: str, text: str) -> bool:
        for alias in self._competency_aliases(name):
            for match in self._alias_pattern(alias).finditer(text):
                if not self._incidental_skill_tail(text, match.end()):
                    return True
        return False

    def _mention_spans(self, name: str, text: str) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        seen = set()
        for alias in self._competency_aliases(name):
            for match in self._alias_pattern(alias).finditer(text):
                key = (match.start(), match.end())
                if key in seen or self._incidental_skill_tail(text, match.end()):
                    continue
                seen.add(key)
                spans.append((match.start(), match.end(), alias))
        spans.sort()
        return spans

    @staticmethod
    def _section_spans(text: str) -> List[Tuple[int, str]]:
        found: List[Tuple[int, str]] = []
        for label, pattern in _SECTION_PATTERNS:
            for match in pattern.finditer(text):
                found.append((match.start(), label))
        found.sort()
        return found

    @staticmethod
    def _section_at(spans: Sequence[Tuple[int, str]], position: int) -> str:
        current = "unknown"
        for start, label in spans:
            if start <= position:
                current = label
            else:
                break
        return current

    @staticmethod
    def _sentence_covering(text: str, start: int, end: int) -> str:
        left_breaks = [text.rfind(mark, 0, start) for mark in (".", "!", "?", "\n", ";")]
        left = max(left_breaks)
        right_breaks = [text.find(mark, end) for mark in (".", "!", "?", "\n", ";")]
        right_breaks = [index for index in right_breaks if index >= 0]
        right = min(right_breaks) if right_breaks else len(text) - 1
        return text[left + 1:right + 1].strip() or text[max(0, start - SECTION_WINDOW):min(len(text), end + SECTION_WINDOW)]

    def _cue_near_skill(
        self,
        pattern: "re.Pattern[str]",
        clause: str,
        alias: str,
        reject_duty: bool = False,
    ) -> bool:
        """A requirement cue counts only when it governs this skill, not a distant clause-mate."""
        for match in self._alias_pattern(alias).finditer(clause):
            start = max(0, match.start() - _CUE_PROXIMITY)
            end = min(len(clause), match.end() + 40)
            window = clause[start:end]
            if reject_duty:
                window = _DUTY_REQUIRED_RE.sub(" ", window)
            if pattern.search(window):
                return True
        return False

    def _clauses_mentioning(self, alias: str, sentence: str) -> List[str]:
        clauses = [
            clause.strip()
            for clause in re.split(r",|;|\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b|\byet\b", sentence, flags=re.I)
            if clause.strip()
        ] or [sentence]
        relevant = [clause for clause in clauses if self._alias_pattern(alias).search(clause)]
        return relevant or clauses[:1]

    def _classify_mentioned_skill(
        self,
        name: str,
        job_text: str,
        section_spans: Optional[Sequence[Tuple[int, str]]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Classify a skill from local JD evidence. None when the text never mentions it."""
        mentions = self._mention_spans(name, job_text)
        if not mentions:
            return None
        spans = list(section_spans) if section_spans is not None else self._section_spans(job_text)
        priorities: List[str] = []
        evidence_for: List[Tuple[str, str]] = []
        generic = self._is_generic_name(name)
        for start, end, alias in mentions:
            sentence = self._sentence_covering(job_text, start, end)
            section = self._section_at(spans, start)
            local = self._requirement_priority(alias, sentence)
            relevant = self._clauses_mentioning(alias, sentence)
            hard = local != "not_required" and any(
                self._cue_near_skill(_HARD_REQUIRED_RE, clause, alias, reject_duty=True)
                for clause in relevant
            )
            experience = local != "not_required" and any(
                self._cue_near_skill(_EXPERIENCE_REQUIRED_RE, clause, alias)
                for clause in relevant
            )
            if local == "not_required":
                chosen = "not_required"
            elif local == "preferred":
                chosen = "preferred"
            elif section == "preferred" and not hard:
                chosen = "preferred"
            elif generic and not hard:
                chosen = "preferred"
            elif section in {"other", "responsibilities"} and not (hard or experience):
                chosen = "preferred"
            elif hard or experience or section == "required":
                chosen = "required"
            else:
                chosen = "required" if local == "required" and not generic else local
            priorities.append(chosen)
            evidence_for.append((chosen, re.sub(r"\s+", " ", sentence).strip()[:240]))
        if "required" in priorities:
            final = "required"
        elif "preferred" in priorities:
            final = "preferred"
        else:
            final = "not_required"
        evidence = next(window for chosen, window in evidence_for if chosen == final)
        return final, evidence[:240]

    def _student_canonical_skills(self, student_text: str, extracted: Sequence[str]) -> set:
        found = {self._canonical_skill(skill) for skill in extracted}
        normalised = _soft_norm(student_text)
        for phrase, canonical in sorted(_ALIAS_LOOKUP.items(), key=lambda item: len(item[0]), reverse=True):
            if len(phrase) < 3 or phrase not in normalised:
                continue
            if self._alias_pattern(phrase).search(normalised):
                found.add(canonical)
        return {skill for skill in found if skill}

    def _skill_match_percent(self, requirements: Sequence[CompetencyRequirement]) -> float:
        """Blend required, preferred and generic coverage so list length cannot dominate.

        Each present tier is normalised on its own. Tier weights then combine
        those coverages: a full preferred tier is worth
        ``preferred_requirement_weight`` relative to a full required tier, and a
        full generic tier is worth ``generic_requirement_weight``. Empty tiers
        drop out and the remaining weights are rescaled, so the result stays
        on 0-100. Explicitly non-required competencies are excluded.
        """
        counts = {"required": [0, 0], "preferred": [0, 0], "generic": [0, 0]}
        for item in requirements:
            if item.priority == "not_required" or item.status == "not_evaluated":
                continue
            if item.priority == "required":
                tier = "required"
            elif self._is_generic_name(item.name):
                tier = "generic"
            else:
                tier = "preferred"
            counts[tier][1] += 1
            if item.status == "matched":
                counts[tier][0] += 1
        weights = {
            "required": 1.0,
            "preferred": self.config.preferred_requirement_weight,
            "generic": self.config.generic_requirement_weight,
        }
        active = {tier: weight for tier, weight in weights.items() if counts[tier][1] > 0}
        if not active:
            return 0.0
        scale = sum(active.values())
        score = 0.0
        for tier, weight in active.items():
            covered, total = counts[tier]
            score += (weight / scale) * (covered / total)
        return max(0.0, min(100.0, score * 100.0))

    def _refresh_assessment(self, st: _State, assessment: _CompetencyAssessment) -> None:
        matched: List[str] = []
        missing: List[str] = []
        learning: List[str] = []
        tiers: Dict[str, int] = {}
        matched_aliases = set()
        for item in assessment.requirements:
            if item.priority == "not_required":
                continue
            name = normalize_skill(item.name)
            if item.status == "matched":
                if name not in matched:
                    matched.append(name)
                matched_aliases.update(self._competency_aliases(item.name))
                continue
            if item.status != "missing":
                continue
            if item.priority == "required":
                tier = 0
                if name not in missing:
                    missing.append(name)
            elif self._is_generic_name(item.name):
                tier = 2
            else:
                tier = 1
            if name not in learning:
                learning.append(name)
                tiers[name] = tier
            else:
                tiers[name] = min(tiers.get(name, tier), tier)
        assessment.requirements.sort(key=lambda item: (item.priority == "not_required", item.name.casefold()))
        assessment.matched = sorted(set(matched), key=str.casefold)
        assessment.missing = sorted(set(missing), key=str.casefold)
        assessment.learning_tiers = tiers
        assessment.learning = self._order_learning(st, learning, tiers)
        assessment.skill_match_percent = self._skill_match_percent(assessment.requirements)
        assessment.matched_aliases = matched_aliases
        assessment.evaluated_count = sum(item.priority == "required" for item in assessment.requirements)

    def _load_or_extract_competencies(self, st: _State) -> List[List[_Competency]]:
        cache = self._competency_cache_file(st)
        if cache.exists():
            try:
                payload = json.loads(cache.read_text(encoding="utf-8"))
                items = payload.get("jobs")
                if isinstance(items, list) and len(items) == len(st.job_ids):
                    return [
                        [
                            _Competency(
                                name=entry["name"],
                                kind=entry["kind"],
                                priority=entry["priority"],
                                evidence=entry["evidence"],
                                generic=bool(entry.get("generic", False)),
                            )
                            for entry in job_entries
                        ]
                        for job_entries in items
                    ]
            except Exception:
                log.warning("Ignoring unreadable competency cache %s", cache)
        extracted = [
            self._extract_competencies(st, description, baseline_skills=sorted(skills))
            for description, skills in zip(st.job_descriptions, st.job_skill_sets)
        ]
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({
                "version": COMPETENCY_LOGIC_VERSION,
                "jobs": [
                    [
                        {
                            "name": item.name,
                            "kind": item.kind,
                            "priority": item.priority,
                            "evidence": item.evidence,
                            "generic": item.generic,
                        }
                        for item in job_entries
                    ]
                    for job_entries in extracted
                ],
            }, ensure_ascii=False), encoding="utf-8")
        except OSError:
            log.warning("Could not write competency cache %s (continuing without it)", cache)
        return extracted

    def _competency_cache_file(self, st: _State) -> Path:
        digest = hashlib.sha256()
        digest.update(COMPETENCY_LOGIC_VERSION.encode())
        digest.update("\n".join(sorted(_soft_norm(item) for item in self.config.generic_competencies)).encode())
        digest.update(repr(_ALIAS_GROUPS).encode())
        for job_id, description, skills in zip(st.job_ids, st.job_descriptions, st.job_skill_sets):
            digest.update(str(job_id).encode())
            digest.update(b"\0")
            digest.update(description.encode())
            digest.update(b"\0")
            digest.update("\n".join(sorted(skills)).encode())
            digest.update(b"\1")
        return Path(self.config.cache_dir) / f"job_competencies_{digest.hexdigest()[:20]}.json"

    def _index_job_skills(self, st: _State) -> None:
        postings: Dict[str, List[int]] = {}
        nongeneric: Dict[str, List[int]] = {}
        canonical_sets: List[set] = []
        for index, (skills, competencies) in enumerate(zip(st.job_skill_sets, st.job_competencies)):
            canonical = {self._canonical_skill(skill) for skill in skills}
            canonical.update(self._canonical_skill(item.name) for item in competencies)
            canonical.discard("")
            canonical_sets.append(canonical)
            for skill in canonical:
                postings.setdefault(skill, []).append(index)
                if not self._is_generic_name(skill):
                    nongeneric.setdefault(skill, []).append(index)
        st.job_canonical_skills = canonical_sets
        st.canonical_postings = {
            key: np.asarray(value, dtype=np.int32) for key, value in postings.items()
        }
        st.nongeneric_postings = {
            key: np.asarray(value, dtype=np.int32) for key, value in nongeneric.items()
        }
        required_terms: List[frozenset] = []
        preferred_terms: List[frozenset] = []
        generic_terms: List[frozenset] = []
        for competencies in st.job_competencies:
            required: set = set()
            preferred: set = set()
            generic: set = set()
            for item in competencies:
                if item.priority == "not_required":
                    continue
                key = self._canonical_skill(item.name)
                if not key:
                    continue
                if item.priority == "required":
                    required.add(key)
                    preferred.discard(key)
                    generic.discard(key)
                elif key in required:
                    continue
                elif item.generic or self._is_generic_name(item.name):
                    generic.add(key)
                    preferred.discard(key)
                else:
                    preferred.add(key)
                    generic.discard(key)
            required_terms.append(frozenset(required))
            preferred_terms.append(frozenset(preferred))
            generic_terms.append(frozenset(generic))
        st.required_terms = required_terms
        st.preferred_terms = preferred_terms
        st.generic_terms = generic_terms

    def _lexical_skill_percents(self, st: _State, student_skills: set) -> np.ndarray:
        """Same tier blend as ``_skill_match_percent``, using canonical overlap only."""
        total = len(st.job_ids)
        scores = np.zeros(total, dtype=np.float64)
        weights = (
            ("required", 1.0, st.required_terms),
            ("preferred", self.config.preferred_requirement_weight, st.preferred_terms),
            ("generic", self.config.generic_requirement_weight, st.generic_terms),
        )
        for index in range(total):
            active_weight = 0.0
            blended = 0.0
            for _name, weight, tiers in weights:
                terms = tiers[index] if index < len(tiers) else frozenset()
                if not terms:
                    continue
                active_weight += weight
                blended += weight * (len(terms & student_skills) / len(terms))
            if active_weight:
                scores[index] = (blended / active_weight) * 100.0
        return scores

    def _select_candidates(
        self,
        st: _State,
        student_skills: set,
        sem: np.ndarray,
        category: np.ndarray,
        full_ranking: bool,
    ) -> List[int]:
        """Retrieve a shortlist without dropping jobs solely for a similarity cutoff."""
        total = len(st.job_ids)
        limit = total if full_ranking else min(self.config.candidate_shortlist_size, total)
        if limit >= total:
            return list(range(total))

        lexical = self._lexical_skill_percents(st, student_skills)
        retrieval = (
            self.config.skill_weight * lexical
            + self.config.semantic_weight * sem
            + self.config.category_weight * category
        )
        nongeneric_hits = np.zeros(total, dtype=np.int16)
        for skill in student_skills:
            if self._is_generic_name(skill):
                continue
            indexes = st.nongeneric_postings.get(skill)
            if indexes is not None and len(indexes):
                nongeneric_hits[indexes] += 1
        skill_slots = min(self.config.retrieval_skill_slots, limit)
        # Role-specific overlaps keep a reserved slice even when semantic
        # similarity is low. Generic-only profiles are ranked by the same
        # hybrid score the final matcher uses, so a matched essential soft
        # skill is not crowded out by a longer metadata list.
        skill_rank = np.argsort(
            -(nongeneric_hits.astype(np.float64) * 1000.0 + lexical),
            kind="stable",
        )
        strong = [int(index) for index in skill_rank if nongeneric_hits[int(index)] > 0][:skill_slots]
        chosen = set(strong)
        selected = list(strong)
        for index in map(int, np.argsort(-retrieval, kind="stable")):
            if index in chosen:
                continue
            selected.append(index)
            chosen.add(index)
            if len(selected) >= limit:
                break
        return selected

    def _is_near_duplicate(self, st: _State, left: int, right: int) -> bool:
        """Collapse near-identical listings without merging different opportunities.

        The same cleaned title is not enough. Category, description and skill
        requirements must also agree before one listing suppresses another.
        """
        if clean_job_title(st.job_titles[left]).lower() != clean_job_title(st.job_titles[right]).lower():
            return False
        if st.job_categories[left] != st.job_categories[right]:
            return False
        if clean_text(st.job_descriptions[left]).lower() == clean_text(st.job_descriptions[right]).lower():
            return True
        cosine = float(np.dot(st.index_vectors[left], st.index_vectors[right]))
        if cosine >= NEAR_DUPLICATE_COSINE:
            return True
        left_skills = st.job_canonical_skills[left] if left < len(st.job_canonical_skills) else set()
        right_skills = st.job_canonical_skills[right] if right < len(st.job_canonical_skills) else set()
        union = len(left_skills | right_skills)
        overlap = len(left_skills & right_skills) / union if union else 1.0
        return overlap >= NEAR_DUPLICATE_SKILL_JACCARD and cosine >= 0.90

    def _apply_batch_semantic_fallback(
        self,
        st: _State,
        assessments: Dict[int, _CompetencyAssessment],
        evidence_vectors: Optional[np.ndarray],
    ) -> None:
        """Paraphrase fallback for the shortlist, in one encoder batch."""
        if (
            st.backend != "sentence-transformers"
            or evidence_vectors is None
            or len(evidence_vectors) == 0
            or self.config.semantic_fallback_max_prompts <= 0
        ):
            return
        pending: List[Tuple[int, int, int, CompetencyRequirement]] = []
        for job_index, assessment in assessments.items():
            for result_index, item in enumerate(assessment.requirements):
                if item.status != "missing" or item.priority == "not_required":
                    continue
                if item.priority != "required" and self._is_generic_name(item.name):
                    continue
                rank = 0 if item.priority == "required" else 1
                pending.append((rank, job_index, result_index, item))
        pending.sort(key=lambda row: (row[0], row[1], row[2]))
        pending = pending[: self.config.semantic_fallback_max_prompts]
        if not pending:
            return
        prompts = [
            f"The candidate has practical knowledge or experience with {item.name}."
            for _, _, _, item in pending
        ]
        similarities = cosine_similarity(self._embed(st, prompts), evidence_vectors)
        changed = set()
        for row, (_, job_index, result_index, _item) in enumerate(pending):
            if float(similarities[row].max()) < COMPETENCY_SEMANTIC_MATCH_THRESHOLD:
                continue
            current = assessments[job_index].requirements[result_index]
            assessments[job_index].requirements[result_index] = current.model_copy(update={"status": "matched"})
            changed.add(job_index)
        for job_index in changed:
            self._refresh_assessment(st, assessments[job_index])

    @staticmethod
    def _is_role_description_mention(skill: str, sentence: str) -> bool:
        """Return True when a known skill is only describing the role/domain."""
        alias = re.escape(normalize_skill(skill))
        role_context = re.compile(
            rf"\b(?:this|the|a|an)\s+(?:[a-z0-9&+.#/-]+\s+){{0,5}}{alias}\s+"
            r"(?:role|position|job|career|opening)\b|"
            rf"\b{alias}\s+(?:role|position|job|opening)\b",
            re.I,
        )
        if not role_context.search(sentence):
            return False

        competency_context = re.compile(
            r"\b(?:require\w*|need\w*|must|mandatory|essential|"
            r"expertise|knowledge|proficiency|experience|skills?|competenc\w*|"
            r"use|using|apply|applying|build|building|implement|implementing|"
            r"develop|developing|work(?:ing)?\s+with|hands[- ]on)\b",
            re.I,
        )
        return not competency_context.search(sentence)

    def _extract_competencies(
        self,
        st: _State,
        job_text: str,
        baseline_skills: Optional[Sequence[str]] = None,
    ) -> List[_Competency]:
        """Extract skills and broader knowledge requirements from arbitrary JD text.

        The dataset vocabulary remains useful for known skills, while generic
        cue phrases discover new topics without a hard-coded subject list.
        """
        candidates: Dict[str, Dict[str, Any]] = {}
        sentences = self._requirement_sentences(job_text)
        section_spans = self._section_spans(job_text)
        baseline_set = set(baseline_skills or [])

        def add(name: str, kind: str, evidence: str, priority: Optional[str] = None) -> None:
            normalized = normalize_skill(name).strip(" .,:;-")
            if not normalized or len(normalized) < 2:
                return
            if priority is None:
                classified = self._classify_mentioned_skill(name, job_text, section_spans)
                if classified is None:
                    priority = self._requirement_priority(normalized, evidence)
                    if self._is_generic_name(name) and priority == "required" and _HARD_REQUIRED_RE.search(evidence) is None:
                        priority = "preferred"
                else:
                    priority, local_evidence = classified
                    evidence = local_evidence or evidence
            canonical = self._canonical_skill(name)
            if not canonical:
                return
            display = name.strip()
            if len(canonical) < len(_soft_norm(display)):
                display = canonical
            current = candidates.get(canonical)
            generic = self._is_generic_name(canonical) or self._is_generic_name(display)
            if current is None:
                candidates[canonical] = {
                    "name": display,
                    "kind": kind,
                    "priorities": [priority],
                    "evidence": evidence[:240],
                    "generic": generic,
                }
                return
            current["priorities"].append(priority)
            if kind == "skill":
                current["kind"] = "skill"
            if len(_soft_norm(display)) < len(_soft_norm(current["name"])):
                current["name"] = display
            if len(evidence) > len(current["evidence"]):
                current["evidence"] = evidence[:240]
            current["generic"] = current["generic"] or generic

        # Metadata skills are candidates. Requirement strength comes from the
        # description, not from the fact that a skill was listed in metadata.
        # User-supplied JDs still discover terms from the text.
        text_skills = [] if baseline_skills is not None else st.skill_extractor.extract(job_text)
        for skill in dict.fromkeys(list(baseline_skills or []) + text_skills):
            matches = [
                sentence for sentence in sentences
                if (baseline_skills is not None or not self._is_role_description_mention(skill, sentence))
                and self._has_substantive_mention(skill, sentence)
            ]
            if not matches and skill in baseline_set:
                add(skill, "skill", _METADATA_EVIDENCE, priority="preferred")
            elif matches:
                add(skill, "skill", matches[0])

        knowledge_cues = re.compile(
            r"\b(?:knowledge|expertise|understanding|proficiency|familiarity|command|grasp|"
            r"mastery|competence|competency)\s+(?:of|in|with|on|for)\s+"
            r"(?P<terms>[^.;:\n]{1,100})",
            re.I,
        )
        experience_cues = re.compile(
            r"\b(?:experience|background|skills?)\s+(?:of|in|with|on|using|applying)\s+"
            r"(?P<terms>[^.;:\n]{1,100})",
            re.I,
        )
        verb_cues = re.compile(
            r"\b(?:must|should|need\s+to|needs\s+to|are\s+required\s+to|is\s+required\s+to)\s+"
            r"(?:know|understand|use|apply|work\s+with|build\s+with|implement\s+with|develop\s+with)\s+"
            r"(?P<terms>[^.;:\n]{1,100})",
            re.I,
        )
        action_cues = re.compile(
            r"\b(?:using|use|apply|applying|work(?:ing)?\s+with|build(?:ing)?\s+with|"
            r"implement(?:ing)?\s+with|develop(?:ing)?\s+with|leverage|utili[sz]e)\s+"
            r"(?P<terms>[^.;:\n]{1,100})",
            re.I,
        )
        list_cues = re.compile(
            r"\b(?:required|preferred|desired|core|nice[- ]to[- ]have)?\s*"
            r"(?:skills?|competencies|tools?|technologies|topics?)\s*"
            r"(?:include|are|:|such\s+as)\s+(?P<terms>[^.;:\n]{1,100})",
            re.I,
        )

        if baseline_skills is None:
            for sentence in sentences:
                for pattern, kind in (
                    (knowledge_cues, "knowledge_area"),
                    (experience_cues, "skill"),
                    (verb_cues, "skill"),
                    (action_cues, "skill"),
                    (list_cues, "skill"),
                ):
                    for match in pattern.finditer(sentence):
                        for term in self._split_requirement_terms(match.group("terms")):
                            known_parts = st.skill_extractor.extract(term)
                            if known_parts and normalize_skill(term) not in {normalize_skill(part) for part in known_parts}:
                                for part in dict.fromkeys(known_parts):
                                    add(part, "skill", sentence)
                            else:
                                add(term, kind, sentence)

        priority_order = {"not_required": 0, "preferred": 1, "required": 2}
        result = []
        for normalized, item in candidates.items():
            priority = max(item["priorities"], key=lambda value: priority_order[value])
            result.append(_Competency(
                name=item["name"],
                kind=item["kind"],
                priority=priority,
                evidence=item["evidence"],
                generic=bool(item.get("generic", False)),
            ))
        return result

    @staticmethod
    def _student_evidence_sentences(student_text: str, records: Sequence[AcademicRecord]) -> List[str]:
        """Split profile evidence and remove academic-grade fragments before skill matching."""
        pieces = re.split(r"(?<=[.!?])\s+|[\r\n;|]+", student_text)
        result = []
        for piece in pieces:
            sentence = piece.strip()
            if not sentence or extract_academic_records(sentence):
                continue
            # Also remove grade evidence embedded in longer unparsed fragments.
            cleaned = sentence
            for record in records:
                if record.scope == "subject" and record.evidence:
                    cleaned = re.sub(re.escape(record.evidence), " ", cleaned, flags=re.I)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-")
            if cleaned:
                result.append(cleaned)
        return result

    @staticmethod
    def _student_mentions_competency(
        name: str,
        evidence_sentences: Sequence[str],
        student_skills: Optional[set] = None,
        evidence_blob: Optional[str] = None,
    ) -> bool:
        """Require evidence of a competency, not merely an aspirational mention."""
        aliases = _cached_competency_aliases(name)
        if evidence_blob is None:
            evidence_blob = "\n".join(evidence_sentences).lower()
        aliases = [alias for alias in aliases if alias in evidence_blob]
        if not aliases:
            return False
        known_skills = {normalize_skill(skill) for skill in (student_skills or set())}
        aspirational = _STUDENT_ASPIRATIONAL_RE
        demonstrated = _STUDENT_DEMONSTRATED_RE
        contrast = _STUDENT_CONTRAST_RE
        list_separator = _STUDENT_LIST_RE

        for sentence in evidence_sentences:
            for clause in contrast.split(sentence):
                clause = clause.strip()
                if not clause:
                    continue
                for alias in aliases:
                    match = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", clause, re.I)
                    if not match:
                        continue
                    start = max(0, match.start() - 55)
                    end = min(len(clause), match.end() + 35)
                    context = clause[start:end]
                    if CareerNLP._span_is_negated(clause, match.start(), match.end()) or aspirational.search(context):
                        continue
                    if demonstrated.search(context):
                        return True
                    # Short extracted terms commonly come from compact skill lists
                    # whose delimiters split the original line. The parser's skill
                    # evidence is accepted only after the negative/aspirational checks.
                    if normalize_skill(alias) in known_skills:
                        return True
                    if list_separator.search(clause) and not re.search(
                        r"\b(?:i|i'm|i am|we|my|our|interested|curious|want|hope|plan|aim|aspire|wish)\b",
                        clause,
                        re.I,
                    ):
                        return True
        return False

    @staticmethod
    def _span_is_negated(text: str, start: int, end: int) -> bool:
        """True when a skill mention is inside a negation such as 'no programming experience'."""
        left_breaks = [text.rfind(mark, 0, start) for mark in (".", "!", "?", "\n", ";")]
        sentence_start = max(left_breaks) + 1
        before = text[max(sentence_start, start - 48):start]
        return _STUDENT_NEGATIVE_RE.search(before) is not None

    def _without_negated_skills(self, text: str, skills: Sequence) -> List:
        """Drop a vocabulary hit when every mention is negated."""
        kept = []
        for skill in skills:
            pattern = re.compile(
                r"(?<!\w)" + re.escape(normalize_skill(skill.name)) + r"(?!\w)",
                re.I,
            )
            spans = list(pattern.finditer(text))
            if spans and all(self._span_is_negated(text, match.start(), match.end()) for match in spans):
                continue
            kept.append(skill)
        return kept

    @staticmethod
    def _semantic_evidence_sentences(sentences: Sequence[str]) -> List[str]:
        """Keep demonstrative profile evidence for semantic matching, not interests."""
        intent = re.compile(
            r"\b(?:interested\s+in|curious\s+about|want(?:s)?\s+to\s+learn|"
            r"hope(?:s)?\s+to\s+learn|plan(?:s)?\s+to\s+learn|aim(?:s)?\s+to\s+learn|"
            r"would\s+like\s+to\s+learn|looking\s+to\s+learn|aspire(?:s)?\s+to|"
            r"wish(?:es)?\s+to\s+learn)\b",
            re.I,
        )
        contrast = re.compile(r"\b(?:but|however|although|whereas|yet)\b", re.I)
        positive = re.compile(
            r"\b(?:skills?|tools?|technologies|competenc(?:y|ies)|proficien\w*|experienced|"
            r"experience|knowledge|expertise|understanding|familiarity|familiar|worked|used|built|"
            r"implemented|developed|applied|created|evaluated|deployed|analysed|analyzed|designed|"
            r"maintained|automated|trained|studied|coursework|certification|certified|internship|"
            r"project|projects|hands[- ]on|practical|demonstrated|programmed|learned|learnt|mastery)\b",
            re.I,
        )
        result: List[str] = []
        for sentence in sentences:
            for clause in contrast.split(sentence):
                clause = clause.strip()
                if not clause:
                    continue
                if intent.search(clause) and not positive.search(clause):
                    continue
                if positive.search(clause) or (
                    re.search(r",|;|\band\b|/|\|", clause)
                    and not re.search(r"\b(?:interested|curious|want|hope|plan|aim|aspire|wish)\b", clause, re.I)
                ):
                    result.append(clause)
        return result

    def _assess_competencies(
        self,
        st: _State,
        job_text: str,
        student_skills: Sequence[str],
        student_evidence_sentences: Sequence[str],
        student_evidence_vectors: Optional[np.ndarray] = None,
        baseline_skills: Optional[Sequence[str]] = None,
        competency_candidates: Optional[Sequence[_Competency]] = None,
        semantic_fallback: bool = True,
    ) -> _CompetencyAssessment:
        """Match extracted requirements and calculate weighted required/preferred coverage."""
        competencies = list(competency_candidates) if competency_candidates is not None else (
            self._extract_competencies(st, job_text, baseline_skills)
        )
        known_student_skills = {normalize_skill(skill) for skill in student_skills}
        evidence_blob = "\n".join(student_evidence_sentences).lower()
        results: List[CompetencyRequirement] = []
        unresolved: List[Tuple[int, _Competency]] = []

        for competency in competencies:
            if competency.priority == "not_required":
                results.append(CompetencyRequirement(
                    name=competency.name,
                    kind=competency.kind,
                    priority=competency.priority,
                    status="not_evaluated",
                    evidence=competency.evidence,
                ))
                continue
            present = self._student_mentions_competency(
                competency.name,
                student_evidence_sentences,
                known_student_skills,
                evidence_blob,
            )
            status = "matched" if present else "missing"
            if status == "missing" and (
                competency.priority == "required" or not self._is_generic_name(competency.name)
            ):
                unresolved.append((len(results), competency))
            results.append(CompetencyRequirement(
                name=competency.name,
                kind=competency.kind,
                priority=competency.priority,
                status=status,
                evidence=competency.evidence,
            ))

        # For paraphrases and acronyms, use the existing Sentence Transformer
        # as a conservative fallback. Explicitly recorded grade fragments were
        # removed from profile evidence, so a low grade cannot count as skill.
        # Generic competencies stay lexical so a broad embedding match cannot
        # turn a soft skill into evidence.
        if (
            unresolved
            and semantic_fallback
            and st.backend == "sentence-transformers"
            and student_evidence_vectors is not None
            and len(student_evidence_vectors) > 0
        ):
            capped = unresolved[: max(self.config.semantic_fallback_max_prompts, 0)]
            if capped:
                prompts = [
                    f"The candidate has practical knowledge or experience with {item.name}."
                    for _, item in capped
                ]
                query_vectors = self._embed(st, prompts)
                similarities = cosine_similarity(query_vectors, student_evidence_vectors)
                for row, (result_index, _competency) in enumerate(capped):
                    if float(similarities[row].max()) < COMPETENCY_SEMANTIC_MATCH_THRESHOLD:
                        continue
                    previous = results[result_index]
                    results[result_index] = previous.model_copy(update={"status": "matched"})

        assessment = _CompetencyAssessment(
            requirements=results,
            matched=[],
            missing=[],
            learning=[],
            skill_match_percent=0.0,
            matched_aliases=set(),
            evaluated_count=0,
        )
        self._refresh_assessment(st, assessment)
        return assessment

    @staticmethod
    def _subject_aliases(subject: str) -> List[str]:
        """Return conservative aliases for common academic subject names."""
        normalized = re.sub(r"[^a-z0-9]+", " ", subject.lower()).strip()
        aliases = {
            "math": ["math", "maths", "mathematics"],
            "maths": ["math", "maths", "mathematics"],
            "mathematics": ["math", "maths", "mathematics"],
            "computer science": ["computer science", "computing"],
            "programming": ["programming", "coding"],
            "physics": ["physics", "physical sciences"],
            "chemistry": ["chemistry", "chemical sciences"],
            "biology": ["biology", "biological sciences"],
            "statistics": ["statistics", "statistical methods"],
            "economics": ["economics"],
            "accounting": ["accounting", "accounts"],
        }
        return aliases.get(normalized, [normalized])

    def _academic_requirement_vectors(
        self, st: _State, subject: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cache positive and neutral requirement examples for semantic matching."""
        key = (st.backend, st.embedder.name, subject.casefold())
        with self._academic_template_lock:
            cached = self._academic_template_cache.get(key)
        if cached is not None:
            return cached

        # Use varied natural-language paraphrases rather than one narrow template.
        # This helps a sentence encoder recognise semantically similar requirements
        # even when the JD avoids the exact terms "strong expertise".
        positive_templates = [
            f"Strong expertise in {subject} is required for this role.",
            f"An advanced understanding of {subject} is essential.",
            f"The candidate must demonstrate deep knowledge of {subject}.",
            f"Expert-level command of {subject} is a core requirement.",
            f"{subject} is a critical competency for this position.",
            f"The position depends on mastery of {subject}.",
            f"Extensive proficiency in {subject} is necessary.",
            f"A thorough grasp of {subject} is a mandatory qualification.",
            f"The role calls for substantial experience applying {subject}.",
            f"Candidates are expected to be highly competent in {subject}.",
            f"Applicants must be adept at applying {subject} to difficult problems.",
            f"The role is built around advanced use of {subject}.",
            f"Day-to-day responsibilities rely heavily on {subject}.",
            f"Success in this position hinges on a strong command of {subject}.",
            f"Candidates need to solve complex problems using {subject} independently.",
            f"A proven track record working with advanced {subject} concepts is required.",
            f"The successful applicant will apply {subject} at an expert level.",
            f"Advanced practical capability in {subject} is central to the job.",
            f"The main responsibility is challenging work involving {subject}.",
            f"Candidates must be highly skilled and experienced in {subject}.",
            f"The role is seeking an expert in {subject}.",
            f"The job requires a specialist who can work independently with {subject}.",
            f"The candidate must have a strong working knowledge of {subject}.",
            f"The job requires more than familiarity; the candidate must master {subject}.",
        ]
        neutral_templates = [
            f"The job description mentions {subject}.",
            f"Basic familiarity with {subject} is sufficient.",
            f"Knowledge of {subject} is optional.",
            f"{subject} is a nice-to-have, not a requirement.",
            f"No expertise in {subject} is required.",
            f"The candidate may encounter {subject} occasionally.",
            f"Some exposure to {subject} is helpful but not essential.",
            f"The role does not depend on prior knowledge of {subject}.",
            f"The candidate can learn {subject} after joining.",
            f"Introductory knowledge of {subject} is enough for this position.",
            f"The work only involves occasional, routine use of {subject}.",
            f"Candidates are not expected to be specialists in {subject}.",
            f"Experience with {subject} is preferred but not mandatory.",
            f"The job can be performed successfully without deep knowledge of {subject}.",
            f"The position is not centered on {subject}.",
            f"{subject} is a minor part of the role, not a core responsibility.",
            f"A candidate can learn the relevant parts of {subject} during onboarding.",
        ]
        vectors = self._embed(st, positive_templates + neutral_templates)
        result = (vectors[:len(positive_templates)], vectors[len(positive_templates):])
        with self._academic_template_lock:
            return self._academic_template_cache.setdefault(key, result)

    def _academic_domain_relevance(
        self,
        st: _State,
        subject: str,
        job_text: str,
    ) -> float:
        """Estimate whether a weak subject is central to the role, on a 0-1 scale.

        Explicit core-domain wording is decisive. Otherwise, the embedding
        backend compares the non-requirement parts of the JD with core-domain
        and incidental-mention examples. Removing requirement-only sentences
        keeps this signal separate from requirement strength. The title is
        deliberately excluded because it can contradict the actual work.
        """
        aliases = sorted(set(self._subject_aliases(subject)), key=len, reverse=True)
        alias_patterns = [
            re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.I)
            for alias in aliases if alias
        ]

        subject_pattern = "(?:" + "|".join(re.escape(a) for a in aliases if a) + ")"
        subject_core_patterns = [
            re.compile(
                subject_pattern
                + r".{0,55}\b(?:is|forms?|serves\s+as|acts\s+as)\b.{0,25}"
                + r"\b(?:the\s+)?(?:core|central|main|primary|principal|dominant|key|critical|"
                + r"major|heart|centre|center|centerpiece|focus)\b",
                re.I,
            ),
            re.compile(
                r"\b(?:core|central|main|primary|principal|dominant|key|critical|major|"
                + r"heart|centre|center|centerpiece)\b.{0,35}\b(?:domain|focus|field|"
                + r"discipline|responsibility|work|speciali[sz]ation)\b.{0,45}"
                + subject_pattern,
                re.I,
            ),
            re.compile(
                r"\b(?:role|position|job|work|responsibilit\w*|tasks?)\b.{0,55}"
                + r"\b(?:built\s+around|centred\s+on|centered\s+on|focused\s+on|"
                + r"focuses\s+on|revolves\s+around|hinges\s+on|depends\s+heavily\s+on)\b"
                + r".{0,45}" + subject_pattern,
                re.I,
            ),
            re.compile(
                subject_pattern
                + r"\s+(?:research|simulations?|engineering|specialists?|speciali[sz]ed|"
                + r"focused|heavy|experiments?)\b",
                re.I,
            ),
        ]
        if any(pattern.search(job_text) for pattern in subject_core_patterns):
            return 1.0

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?;])\s+|[\r\n]+", job_text)
            if sentence.strip()
        ]
        requirement_only = re.compile(
            r"\b(?:expertise|proficiency|knowledge|understanding|academic\s+foundation|"
            r"strong\s+background|qualification|mandatory|required|requirement|must\s+have|"
            r"essential|prerequisite|aptitude|competenc\w*)\b",
            re.I,
        )
        domain_sentences = [
            sentence for sentence in sentences
            if not (requirement_only.search(sentence) and any(p.search(sentence) for p in alias_patterns))
        ]
        domain_text = " ".join(domain_sentences).strip()
        if not domain_text:
            return 0.0

        # Cache semantic examples by subject and backend, just like the existing
        # requirement examples. No extra model or API dependency is introduced.
        key = (st.backend, st.embedder.name, f"domain:{subject.casefold()}")
        with self._academic_template_lock:
            cached = self._academic_template_cache.get(key)
        if cached is None:
            positive_templates = [
                f"The role's central technical work is in {subject}.",
                f"The main responsibilities involve applying {subject} concepts directly.",
                f"This is a specialist position in the {subject} field.",
                f"Most projects in this job are within the {subject} domain.",
                f"The job involves {subject} research, experiments, or methods as regular work.",
                f"The candidate applies {subject} to the role's main technical problems.",
                f"The position develops solutions grounded in {subject} concepts.",
                f"The role's day-to-day work depends on methods from {subject}.",
            ]
            neutral_templates = [
                f"The role is a general business or data position unrelated to {subject}.",
                f"The job mentions {subject} only as background or occasional context.",
                f"The position focuses on another field; {subject} is incidental.",
                f"The role uses general tools and does not apply {subject} concepts.",
                f"The job handles data about {subject} without doing subject-specific work.",
                f"The main tasks are general reporting and data cleaning, not {subject}.",
            ]
            vectors = self._embed(st, positive_templates + neutral_templates)
            cached = (vectors[:len(positive_templates)], vectors[len(positive_templates):])
            with self._academic_template_lock:
                cached = self._academic_template_cache.setdefault(key, cached)

        positive_vectors, neutral_vectors = cached
        context_vector = self._embed(st, [domain_text])
        positive_score = float(cosine_similarity(context_vector, positive_vectors).max())
        neutral_score = float(cosine_similarity(context_vector, neutral_vectors).max())
        margin = positive_score - neutral_score
        if (
            positive_score < ACADEMIC_DOMAIN_MIN_SIMILARITY
            or margin <= ACADEMIC_DOMAIN_MIN_MARGIN
        ):
            return 0.0

        relevance = (margin - ACADEMIC_DOMAIN_MIN_MARGIN) / ACADEMIC_DOMAIN_MARGIN_RANGE
        return round(min(1.0, max(0.0, relevance)), 2)

    def _semantic_requirement_supported(
        self, st: _State, subject: str, clause: str
    ) -> bool:
        """Compare a JD clause with strong-requirement and neutral examples."""
        positive_vectors, neutral_vectors = self._academic_requirement_vectors(st, subject)
        clause_vector = self._embed(st, [clause])
        positive_score = float(cosine_similarity(clause_vector, positive_vectors).max())
        neutral_score = float(cosine_similarity(clause_vector, neutral_vectors).max())
        # Requiring both an absolute score and a margin avoids treating any
        # simple mention of a subject as an expertise requirement.
        return (
            positive_score >= ACADEMIC_REQUIREMENT_MIN_SIMILARITY
            and positive_score >= neutral_score + ACADEMIC_REQUIREMENT_MIN_MARGIN
        )

    def _academic_adjustments(
        self,
        st: _State,
        records: Sequence[AcademicRecord],
        job_text: str,
        demonstrated_competencies: Optional[set] = None,
    ) -> List[AcademicAdjustment]:
        """Penalise poor subject grades only when the JD strongly requires it.

        Explicit rules and negation checks are deterministic. Embedding-based
        comparison is the fallback for varied requirement wording. Overall GPAs
        are reported to the caller but are not mapped to a subject.
        """
        poor_records = [
            record for record in records
            if record.is_poor and record.scope == "subject"
        ]
        if not poor_records:
            return []

        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?;])\s+|[\r\n]+", job_text)
            if part.strip()
        ] or [job_text]
        strong_requirement = re.compile(
            r"\b(?:strong|advanced|deep|expert(?:[- ]level)?|high|excellent|extensive|"
            r"substantial|solid|sound|thorough|comprehensive|exceptional|profound|firm|"
            r"speciali[sz]ed|in[- ]depth)\b.{0,75}\b(?:expertise|knowledge|proficiency|"
            r"skills?|understanding|background|experience|command|grasp|mastery|"
            r"competenc(?:e|y)|ability|aptitude)\b"
            r"|\b(?:mastery|proficiency|expertise|command|grasp)\b.{0,40}\b(?:of|in|with)\b"
            r"|\b(?:adept|skilled|proficient|knowledgeable|fluent|well[- ]versed|competent)\b"
            r".{0,45}\b(?:at|in|with|applying|using)\b"
            r"|\b(?:built\s+around|cent(?:er|re)s?\s+on|revolves?\s+around|"
            r"hinges?\s+on|depends?\s+heavily\s+on|relies?\s+heavily\s+on)\b"
            r"|\b(?:advanced|complex|challenging|speciali[sz]ed)\s+(?:work|tasks?|problems?|projects?)\b"
            r".{0,45}\b(?:involving|using|in|with)\b",
            re.I,
        )
        required_requirement = re.compile(
            r"\b(?:require\w*|need\w*|must\s+(?:have|know|understand|demonstrate|possess|apply|use|solve)|"
            r"essential|mandatory|necessary|indispensable|prerequisite|critical|crucial|"
            r"central|fundamental|foundational|pivotal|core|primary|main|expected|demonstrate|"
            r"competent|calls\s+for|depends\s+on|responsibilit(?:y|ies)|main\s+focus)\b",
            re.I,
        )
        negative_requirement = re.compile(
            r"\b(?:does\s+not|doesn't|do\s+not|don't|not(?!\s+only)|no|without|isn't|is\s+not|"
            r"never|need\s+not|not\s+necessarily)\b.{0,85}\b(?:require\w*|need\w*|"
            r"expertise|knowledge|proficiency|skills?|understanding|background|experience|"
            r"essential|mandatory|prerequisite|necessary)\b",
            re.I,
        )
        requirement_not_needed = re.compile(
            r"\b(?:expertise|knowledge|proficiency|skills?|understanding|background|experience|"
            r"mastery|competenc(?:e|y))\b.{0,65}\b(?:is\s+)?(?:not\s+required|not\s+needed|"
            r"not\s+essential|not\s+necessary|unnecessary|not\s+a\s+prerequisite)\b",
            re.I,
        )
        optional_requirement = re.compile(
            r"\b(?:nice[- ]to[- ]have|optional|a\s+plus|bonus|desirable|preferable|preferred|"
            r"not\s+mandatory|not\s+essential|not\s+necessary|basic\s+familiarity\s+is\s+"
            r"sufficient|helpful|useful|beneficial|advantageous|can\s+be\s+learned)\b",
            re.I,
        )
        not_core_requirement = re.compile(
            r"\b(?:not(?!\s+only)|isn't|is\s+not|doesn't|does\s+not|never)\b"
            r".{0,80}\b(?:core|central|main|primary|key|critical|essential|focus|"
            r"built\s+around|cent(?:er|re)(?:ed|red|s)?\s+on|revolves?\s+around|"
            r"depends?\s+on|hinges?\s+on|main\s+responsibilit(?:y|ies))\b",
            re.I,
        )

        adjustments: List[AcademicAdjustment] = []
        seen_subjects = set()
        for record in poor_records:
            subject_key = re.sub(r"[^a-z0-9]+", " ", record.subject.casefold()).strip()
            if subject_key in seen_subjects:
                continue
            aliases = sorted(set(self._subject_aliases(record.subject)), key=len, reverse=True)
            if demonstrated_competencies and any(
                re.search(
                    r"(?<!\w)" + re.escape(normalize_skill(alias)) + r"(?!\w)",
                    demonstrated_name,
                    re.I,
                )
                for alias in aliases
                for demonstrated_name in demonstrated_competencies
            ):
                # A poor academic result is a risk signal, not proof of a
                # current competency gap. Avoid penalising a subject when the
                # profile independently demonstrates the required knowledge.
                continue
            # These patterns link the expertise wording directly to the weak
            # subject, covering forms such as "an expert in Physics" and
            # "Physics specialist" without treating any unrelated expert role
            # in the same sentence as evidence about Physics.
            subject_expertise_patterns = []
            for alias in aliases:
                alias_pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
                subject_expertise_patterns.extend([
                    re.compile(
                        r"\b(?:expert|specialist|authority|subject[- ]matter expert)\b"
                        r".{0,35}\b(?:in|of|on|with)\s*" + alias_pattern,
                        re.I,
                    ),
                    re.compile(alias_pattern + r"\s+(?:expert|specialist|authority)\b", re.I),
                    re.compile(
                        r"\b(?:strong|advanced|deep|excellent|comprehensive|thorough|solid|good)\s+"
                        r"(?:working\s+)?(?:knowledge|understanding|command|grasp|proficiency|mastery)\s+"
                        r"(?:of|in|with)\s*" + alias_pattern,
                        re.I,
                    ),
                ])
            matched_reason: Optional[str] = None
            subject_excluded = False

            for sentence in sentences:
                # Keep contrasting clauses separate: a negative elsewhere in a
                # sentence should not cancel a later, affirmative requirement.
                clauses = [
                    clause.strip()
                    for clause in re.split(
                        r",|\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b|\byet\b",
                        sentence,
                        flags=re.I,
                    )
                    if clause.strip()
                ] or [sentence]
                for clause in clauses:
                    if not any(
                        re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", clause, re.I)
                        for alias in aliases if alias
                    ):
                        continue

                    if (
                        negative_requirement.search(clause)
                        or requirement_not_needed.search(clause)
                        or optional_requirement.search(clause)
                        or not_core_requirement.search(clause)
                    ):
                        subject_excluded = True
                        continue

                    if any(pattern.search(clause) for pattern in subject_expertise_patterns):
                        matched_reason = "explicit subject-expertise wording"
                    elif strong_requirement.search(clause):
                        matched_reason = "explicit strong-expertise wording"
                    elif required_requirement.search(clause):
                        matched_reason = "explicit mandatory-requirement wording"
                    elif self._semantic_requirement_supported(st, record.subject, clause):
                        matched_reason = "semantic comparison with strong-requirement examples"
                    if matched_reason:
                        break
                if matched_reason:
                    break

            domain_relevance: Optional[float] = None
            if not matched_reason and not subject_excluded:
                domain_relevance = self._academic_domain_relevance(
                    st, record.subject, job_text
                )
                if domain_relevance >= 0.75:
                    matched_reason = "core-domain evidence from the job description and responsibilities"

            if matched_reason:
                seen_subjects.add(subject_key)
                if domain_relevance is None:
                    domain_relevance = self._academic_domain_relevance(
                        st, record.subject, job_text
                    )
                base_penalty = self.config.academic_subject_penalty
                core_penalty = self.config.academic_core_domain_penalty
                penalty = round(
                    base_penalty + (core_penalty - base_penalty) * domain_relevance,
                    2,
                )
                if record.score_percent is not None:
                    grade_evidence = f"{record.score_percent:g}%"
                    grade_reason = (
                        f"the recorded score is {grade_evidence}, below the "
                        f"{self.config.poor_grade_threshold_percent:g}% poor-performance threshold"
                    )
                else:
                    grade_evidence = record.grade or "poor academic performance"
                    grade_reason = f"the recorded grade is {grade_evidence}, flagged as poor"

                if domain_relevance >= 0.5:
                    reason = (
                        f"{record.subject} is a core technical domain of this role "
                        f"(domain relevance {domain_relevance * 100:.0f}%), and {grade_reason}. "
                        f"The JD also contains {matched_reason}."
                    )
                else:
                    reason = (
                        f"The JD strongly requires {record.subject}, while {grade_reason}. "
                        f"Core-domain relevance is {domain_relevance * 100:.0f}%; "
                        f"requirement evidence: {matched_reason}."
                    )
                adjustments.append(AcademicAdjustment(
                    subject=record.subject,
                    grade=record.grade,
                    score_percent=record.score_percent,
                    scale_max=record.scale_max,
                    domain_relevance_percent=round(domain_relevance * 100.0, 2),
                    penalty_percent=penalty,
                    reason=reason,
                ))
        return adjustments

    @staticmethod
    def _category_probability_percent(st: _State, student_text: str, category: str) -> float:
        """Return the profile classifier's probability for ``category`` on 0-100.

        It is deliberately a modest reranking signal: exact skill evidence and
        semantic similarity remain the dominant parts of each match score.
        """
        proba = st.classifier.predict_proba(
            st.classifier_vectorizer.transform([clean_text(student_text)])
        )[0]
        by_category = dict(zip(st.classifier.classes_, proba))
        return float(by_category.get(category, 0.0) * 100.0)

    # ------------------------------------------------------------------
    # information extraction
    # ------------------------------------------------------------------
    def _extract_fields(self, st: _State, text: str) -> Dict[str, Any]:
        # Academic parsing needs original line breaks because CVs often list
        # one subject/grade per line. Other extractors use normalised text.
        academic_records = extract_academic_records(
            text, poor_threshold_percent=self.config.poor_grade_threshold_percent
        )
        text = clean_text(text)
        skills = self._without_negated_skills(text, extract_skills(text, st.skill_extractor))
        degrees = extract_degrees(text)
        emails = extract_emails(text)
        phones = extract_phones(text)
        mentions, years = extract_experience(text)
        return dict(
            skills=skills, degrees_or_certifications=degrees, emails=emails,
            phone_numbers=phones, experience_mentions=mentions, experience_years=years,
            academic_records=academic_records,
            entities_count=(len(skills) + len(degrees) + len(emails) + len(phones)
                            + len(mentions) + len(academic_records)),
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
            academic_records=fields.get("academic_records", []),
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
    def _match_from_state(
        self,
        st: _State,
        student_text: str,
        job_text: str,
        include_details: bool,
        t0: float,
        fields: Optional[Dict[str, Any]] = None,
        profile: Optional[StudentProfile] = None,
        student_vector: Optional[np.ndarray] = None,
        job_vector: Optional[np.ndarray] = None,
        job_title: Optional[str] = None,
        student_evidence_sentences: Optional[Sequence[str]] = None,
        student_evidence_vectors: Optional[np.ndarray] = None,
        baseline_skills: Optional[Sequence[str]] = None,
    ) -> MatchResponse:
        fields = fields or self._extract_fields(st, student_text)
        profile = profile or self._profile_from_fields(fields)
        student_skills = set(profile.skills)
        evidence_sentences = list(student_evidence_sentences or self._student_evidence_sentences(
            student_text, profile.academic_records
        ))
        semantic_evidence = self._semantic_evidence_sentences(evidence_sentences)
        if student_evidence_vectors is None and st.backend == "sentence-transformers" and semantic_evidence:
            student_evidence_vectors = self._embed(st, semantic_evidence)
        assessment = self._assess_competencies(
            st,
            job_text,
            list(student_skills),
            evidence_sentences,
            student_evidence_vectors,
            baseline_skills=baseline_skills,
        )
        competency_requirements = assessment.requirements
        matched = assessment.matched
        missing = assessment.missing
        skill_pct = assessment.skill_match_percent
        matched_competency_aliases = assessment.matched_aliases
        n_requirements = assessment.evaluated_count

        # Keep professional context in the embedding, and let batch comparison
        # pass precomputed vectors so the same CV is not embedded repeatedly.
        semantic_profile = f"{clean_text(student_text)} {profile.profile_summary}"
        scoring_job_text = (
            f"Job title: {job_title}. Job description: {job_text}"
            if job_title else job_text
        )
        sv = student_vector if student_vector is not None else self._embed(st, [semantic_profile])
        jv = job_vector if job_vector is not None else self._embed(st, [scoring_job_text])
        sem_pct = float(self._cos_percent(sv, jv)[0])
        job_category = str(st.classifier.predict(
            st.classifier_vectorizer.transform([clean_text(scoring_job_text)])
        )[0])
        category_pct = self._category_probability_percent(st, student_text, job_category)

        base_score = round(self._combine(skill_pct, sem_pct, category_pct), 2)
        adjustments = self._academic_adjustments(
            st,
            profile.academic_records,
            job_text,
            demonstrated_competencies=matched_competency_aliases,
        )
        academic_penalty = round(min(100.0, sum(a.penalty_percent for a in adjustments)), 2)
        score = round(max(0.0, base_score - academic_penalty), 2)

        warnings: List[str] = []
        if n_requirements == 0 and not any(item.priority == "preferred" for item in competency_requirements):
            warnings.append(
                "No known skills were recognised in the job description, and no broader competencies were "
                "identified, so Skill Match is 0 and the score relies only on semantic and category signals."
            )
        if not student_skills and not matched:
            warnings.append("No known skills were recognised in the student text; Skill Match is 0.")
        if adjustments:
            subjects = ", ".join(a.subject for a in adjustments)
            warnings.append(f"Academic adjustment applied for poor grades in required subject(s): {subjects}.")

        confidence = conf.match_confidence(
            n_requirements, len(student_skills), include_factors=include_details
        )
        return MatchResponse(
            match_score=score,
            base_match_score=base_score,
            academic_penalty_percent=academic_penalty,
            academic_adjustments=adjustments,
            academic_records=profile.academic_records,
            competency_requirements=competency_requirements,
            matched_skills=matched,
            missing_weak_skills=missing,
            recommended_learning=list(assessment.learning),
            confidence=confidence,
            score_breakdown=(
                self._breakdown(skill_pct, sem_pct, category_pct, n_requirements, base_score, academic_penalty)
                if include_details else None
            ),
            student_profile=profile if include_details else None,
            meta=self._meta(st, t0, warnings),
        )

    def match(self, student_text: str, job_text: str, include_details: bool = False) -> MatchResponse:
        t0 = time.perf_counter()
        self._check_text(student_text, "student_text")
        self._check_text(job_text, "job_text")
        st = self._ensure()
        fields = self._extract_fields(st, student_text)
        profile = self._profile_from_fields(fields)
        return self._match_from_state(
            st, student_text, job_text, include_details, t0, fields=fields, profile=profile
        )

    def compare_jobs(
        self,
        student_text: str,
        job_descriptions: Sequence[JobDescriptionInput],
        include_details: bool = False,
    ) -> CompareJobsResponse:
        """Rank two or more caller-supplied JDs against one student profile."""
        t0 = time.perf_counter()
        self._check_text(student_text, "student_text")
        if not isinstance(job_descriptions, (list, tuple)) or not 2 <= len(job_descriptions) <= 20:
            raise InputValidationError(
                "'job_descriptions' must contain between 2 and 20 job descriptions.",
                {"field": "job_descriptions", "min": 2, "max": 20},
            )

        candidates: List[JobDescriptionInput] = []
        for index, item in enumerate(job_descriptions, start=1):
            if isinstance(item, JobDescriptionInput):
                candidate = item
            elif isinstance(item, dict):
                try:
                    candidate = JobDescriptionInput.model_validate(item)
                except Exception as exc:
                    raise InputValidationError(
                        f"Invalid job description at position {index}.",
                        {"field": f"job_descriptions[{index - 1}]", "error": str(exc)},
                    ) from exc
            else:
                raise InputValidationError(
                    f"Job description at position {index} must be an object.",
                    {"field": f"job_descriptions[{index - 1}]"},
                )
            self._check_text(candidate.job_text, f"job_descriptions[{index - 1}].job_text")
            candidates.append(candidate)

        st = self._ensure()
        fields = self._extract_fields(st, student_text)
        profile = self._profile_from_fields(fields)
        semantic_profile = f"{clean_text(student_text)} {profile.profile_summary}"
        evidence_sentences = self._student_evidence_sentences(student_text, profile.academic_records)
        scoring_job_texts = [
            f"Job title: {item.job_title or f'Job description {index}'}. Job description: {item.job_text}"
            for index, item in enumerate(candidates, start=1)
        ]
        texts_to_embed = [semantic_profile] + scoring_job_texts
        semantic_evidence = self._semantic_evidence_sentences(evidence_sentences)
        if st.backend == "sentence-transformers":
            texts_to_embed.extend(semantic_evidence)
        # Batch-encode title + body, the student profile and evidence sentences
        # once so multi-JD comparison does not repeatedly run the encoder.
        vectors = self._embed(st, texts_to_embed)
        evidence_vectors = (
            vectors[1 + len(candidates):]
            if st.backend == "sentence-transformers" and semantic_evidence else None
        )
        compared: List[ComparedJob] = []
        used_ids = set()
        for index, item in enumerate(candidates, start=1):
            job_id = item.job_id or f"JD{index}"
            if job_id in used_ids:
                raise InputValidationError(
                    "Each job description must have a unique job_id when supplied.",
                    {"field": "job_descriptions", "duplicate_job_id": job_id},
                )
            used_ids.add(job_id)
            job_title = item.job_title or f"Job description {index}"
            match_result = self._match_from_state(
                st, student_text, item.job_text, include_details, t0,
                fields=fields, profile=profile,
                student_vector=vectors[0:1], job_vector=vectors[index:index + 1],
                job_title=job_title,
                student_evidence_sentences=evidence_sentences,
                student_evidence_vectors=evidence_vectors,
            )
            compared.append(ComparedJob(
                rank=0, job_id=job_id, job_title=job_title, match=match_result
            ))

        compared.sort(key=lambda item: item.match.match_score, reverse=True)
        ranked = [item.model_copy(update={"rank": rank}) for rank, item in enumerate(compared, start=1)]
        return CompareJobsResponse(
            recommended_job=ranked[0],
            alternatives=ranked[1:],
            academic_records=profile.academic_records,
            meta=self._meta(st, t0),
        )

    # ------------------------------------------------------------------
    # recommendation - JD not provided
    # ------------------------------------------------------------------
    def _job_record(
        self,
        st: _State,
        i: int,
        rank: int,
        sem_pct: float,
        category_pct: float,
        academic_records: Sequence[AcademicRecord],
        details: bool,
        competency_assessment: _CompetencyAssessment,
    ) -> Tuple[RecommendedJob, float]:
        competency_requirements = competency_assessment.requirements
        matched = competency_assessment.matched
        missing = competency_assessment.missing
        skill_pct = competency_assessment.skill_match_percent
        matched_aliases = competency_assessment.matched_aliases
        n_requirements = competency_assessment.evaluated_count
        base_score = round(self._combine(skill_pct, sem_pct, category_pct), 2)
        adjustments = self._academic_adjustments(
            st,
            academic_records,
            st.job_descriptions[i],
            demonstrated_competencies=matched_aliases,
        )
        academic_penalty = round(min(100.0, sum(a.penalty_percent for a in adjustments)), 2)
        score = round(max(0.0, base_score - academic_penalty), 2)
        rec = RecommendedJob(
            rank=rank,
            job_id=st.job_ids[i],
            job_title=clean_job_title(st.job_titles[i]),
            category=st.job_categories[i],
            match_score=score,
            base_match_score=base_score,
            academic_penalty_percent=academic_penalty,
            academic_adjustments=adjustments,
            competency_requirements=competency_requirements,
            matched_skills=matched,
            missing_weak_skills=missing,
            recommended_learning=list(competency_assessment.learning),
            score_breakdown=(
                self._breakdown(
                    skill_pct, sem_pct, category_pct, n_requirements, base_score, academic_penalty
                ) if details else None
            ),
        )
        return rec, skill_pct

    def recommend(
        self,
        student_text: str,
        top_k: int = 3,
        include_details: bool = False,
        *,
        full_ranking: bool = False,
    ) -> RecommendResponse:
        t0 = time.perf_counter()
        self._check_text(student_text, "student_text")
        k = self._top_k(top_k, 3, 10)
        st = self._ensure()

        profile = self._profile_from_fields(self._extract_fields(st, student_text))
        student_skills = set(profile.skills)
        evidence_sentences = self._student_evidence_sentences(student_text, profile.academic_records)
        semantic_profile = f"{clean_text(student_text)} {profile.profile_summary}"
        sem = self._cos_percent(self._embed(st, [semantic_profile]), st.index_vectors)
        category_proba = st.classifier.predict_proba(
            st.classifier_vectorizer.transform([clean_text(student_text)])
        )[0]
        category_lookup = dict(zip(st.classifier.classes_, category_proba))
        category = np.array(
            [category_lookup.get(job_category, 0.0) * 100.0 for job_category in st.job_categories]
        )
        evidence_blob = "\n".join(evidence_sentences).lower()
        canonical_student = {
            skill
            for skill in self._student_canonical_skills(student_text, profile.skills)
            if self._student_mentions_competency(
                skill, evidence_sentences, student_skills, evidence_blob
            )
        }
        selected = self._select_candidates(st, canonical_student, sem, category, full_ranking)
        competency_assessments: Dict[int, _CompetencyAssessment] = {
            i: self._assess_competencies(
                st,
                st.job_descriptions[i],
                list(student_skills),
                evidence_sentences,
                baseline_skills=sorted(st.job_skill_sets[i]),
                competency_candidates=st.job_competencies[i],
                semantic_fallback=False,
            )
            for i in selected
        }
        semantic_evidence = self._semantic_evidence_sentences(evidence_sentences)
        evidence_vectors = (
            self._embed(st, semantic_evidence)
            if st.backend == "sentence-transformers" and semantic_evidence else None
        )
        self._apply_batch_semantic_fallback(st, competency_assessments, evidence_vectors)
        scores: Dict[int, float] = {}
        for i in selected:
            skill_pct = competency_assessments[i].skill_match_percent
            base_score = round(self._combine(skill_pct, float(sem[i]), float(category[i])), 2)
            penalty = 0.0
            if profile.academic_records:
                adjustments = self._academic_adjustments(
                    st,
                    profile.academic_records,
                    st.job_descriptions[i],
                    demonstrated_competencies=competency_assessments[i].matched_aliases,
                )
                penalty = min(100.0, sum(item.penalty_percent for item in adjustments))
            scores[i] = round(max(0.0, base_score - penalty), 2)
        order = sorted(selected, key=lambda index: (-scores[index], index))

        # Drop near-identical listings. The same cleaned title can still appear
        # when the category or the description/requirements are materially different.
        picked: List[int] = []
        for i in order:
            if any(self._is_near_duplicate(st, i, previous) for previous in picked):
                continue
            picked.append(i)
            if len(picked) == k:
                break

        records, top_skill_pct = [], 0.0
        for rank, i in enumerate(picked, start=1):
            rec, skill_pct = self._job_record(
                st, i, rank, float(sem[i]), float(category[i]),
                profile.academic_records, include_details,
                competency_assessment=competency_assessments[i],
            )
            records.append(rec)
            if rank == 1:
                top_skill_pct = skill_pct

        top_slice = order[:5]
        top5 = [st.job_categories[i] for i in top_slice]
        agreement = top5.count(st.job_categories[order[0]]) / len(top5)
        second = scores[order[1]] if len(order) > 1 else None
        confidence = conf.recommend_confidence(
            top_score=scores[order[0]], second_score=second,
            top_skill_match_percent=top_skill_pct, category_agreement=agreement,
            n_student_skills=len(student_skills), include_factors=include_details)
        self.last_recommendation_stats = {
            "jobs_indexed": len(st.job_ids),
            "shortlist_size": len(selected),
            "jobs_assessed": len(selected),
            "full_ranking": bool(full_ranking or len(selected) == len(st.job_ids)),
            "assessed_job_ids": [st.job_ids[i] for i in selected],
        }

        warnings: List[str] = []
        if not student_skills:
            warnings.append("No known skills were recognised in the student text; the recommendation relies on "
                            "semantic similarity alone and is unreliable.")
        if confidence.level == "low" and student_skills:
            warnings.append("Low confidence: top results are close in score and/or span several job categories.")

        if records[0].academic_adjustments:
            subjects = ", ".join(a.subject for a in records[0].academic_adjustments)
            warnings.append(f"Top recommendation includes an academic adjustment for: {subjects}.")

        return RecommendResponse(
            recommended_job=records[0], alternatives=records[1:], confidence=confidence,
            academic_records=profile.academic_records,
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
            scoring={
                "skill_weight": self.config.skill_weight,
                "semantic_weight": self.config.semantic_weight,
                "category_weight": self.config.category_weight,
                "poor_grade_threshold_percent": self.config.poor_grade_threshold_percent,
                "academic_subject_penalty": self.config.academic_subject_penalty,
                "academic_core_domain_penalty": self.config.academic_core_domain_penalty,
                "preferred_requirement_weight": self.config.preferred_requirement_weight,
                "generic_requirement_weight": self.config.generic_requirement_weight,
            },
            limits={"max_text_chars": self.config.max_text_chars, "max_cv_bytes": self.config.max_cv_bytes,
                    "max_top_k": self.config.max_top_k,
                    "candidate_shortlist_size": self.config.candidate_shortlist_size,
                    "retrieval_skill_slots": self.config.retrieval_skill_slots},
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
