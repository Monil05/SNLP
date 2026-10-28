"""CareerNLP - the reusable engine.

One object, lazily initialised, thread-safe after warm-up, returning validated
Pydantic results. It owns: job dataset + skill vocabulary, category classifier,
embedding backend + job index (cached on disk), and the match/recommend logic.
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


@dataclass(frozen=True)
class _Competency:
    name: str
    kind: str
    priority: str
    evidence: str


@dataclass(frozen=True)
class _CompetencyAssessment:
    requirements: List[CompetencyRequirement]
    matched: List[str]
    missing: List[str]
    learning: List[str]
    skill_match_percent: float
    matched_aliases: set
    evaluated_count: int


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
        state.job_competencies = [
            self._extract_competencies(state, description, baseline_skills=sorted(skills))
            for description, skills in zip(state.job_descriptions, state.job_skill_sets)
        ]
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

    def _order_learning(self, st: _State, missing: Sequence[str]) -> List[str]:
        """Most in-demand (across the job dataset) first, then alphabetical."""
        return sorted(missing, key=lambda s: (-st.skill_demand.get(s, 0), s))

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
    def _competency_aliases(name: str) -> List[str]:
        """Return phrase/acronym aliases inferred from the phrase itself."""
        aliases = {normalize_skill(name)}
        without_parentheses = re.sub(r"\s*\([^)]*\)", "", name).strip()
        if without_parentheses:
            aliases.add(normalize_skill(without_parentheses))
        for acronym in re.findall(r"\(([A-Z][A-Z0-9+#.-]{1,9})\)", name):
            aliases.add(normalize_skill(acronym))
        return sorted((alias for alias in aliases if alias), key=len, reverse=True)

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

        def add(name: str, kind: str, evidence: str, from_metadata: bool = False) -> None:
            normalized = normalize_skill(name).strip(" .,:;-")
            if not normalized or len(normalized) < 2:
                return
            priority = "required" if from_metadata else self._requirement_priority(normalized, evidence)
            current = candidates.get(normalized)
            if current is None:
                candidates[normalized] = {
                    "name": name.strip(),
                    "kind": kind,
                    "priorities": [priority],
                    "evidence": evidence[:240],
                }
                return
            current["priorities"].append(priority)
            if kind == "skill":
                current["kind"] = "skill"
            if len(evidence) > len(current["evidence"]):
                current["evidence"] = evidence[:240]

        # When using the bundled job dataset, its explicit skill-set column is
        # already curated. For user-supplied JDs, discover terms from the text.
        text_skills = [] if baseline_skills is not None else st.skill_extractor.extract(job_text)
        for skill in dict.fromkeys(list(baseline_skills or []) + text_skills):
            matches = [
                sentence for sentence in sentences
                if (baseline_skills is not None or not self._is_role_description_mention(skill, sentence))
                and any(
                    re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", sentence, re.I)
                    for alias in self._competency_aliases(skill)
                )
            ]
            if not matches and baseline_skills and skill in baseline_skills:
                add(skill, "skill", "Curated job skill metadata.", from_metadata=True)
            else:
                for evidence in matches or [job_text]:
                    add(skill, "skill", evidence)

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
    ) -> bool:
        """Require evidence of a competency, not merely an aspirational mention."""
        aliases = CareerNLP._competency_aliases(name)
        known_skills = {normalize_skill(skill) for skill in (student_skills or set())}
        negative = re.compile(
            r"\b(?:no\s+(?:prior\s+)?(?:knowledge|experience|familiarity|proficiency|skills?)|"
            r"lack(?:s|ing)?\s+(?:of\s+)?(?:knowledge|experience|familiarity|proficiency|skills?)|"
            r"not\s+(?:familiar|experienced|proficient|skilled|knowledgeable)\s+(?:with|in|about)|"
            r"never\s+used|haven't\s+used|have\s+not\s+used|don't\s+know|do\s+not\s+know|"
            r"not\s+yet\s+(?:used|learned|learnt|studied|worked\s+with))\b",
            re.I,
        )
        aspirational = re.compile(
            r"\b(?:interested\s+in|curious\s+about|want(?:s)?\s+to\s+learn|"
            r"hope(?:s)?\s+to\s+learn|plan(?:s)?\s+to\s+learn|aim(?:s)?\s+to\s+learn|"
            r"would\s+like\s+to\s+learn|looking\s+to\s+learn|aspire(?:s)?\s+to|"
            r"wish(?:es)?\s+to\s+learn)\b",
            re.I,
        )
        demonstrated = re.compile(
            r"\b(?:skills?|tools?|technologies|competenc(?:y|ies)|proficien\w*|experienced|"
            r"experience|knowledge|expertise|understanding|familiarity|familiar|worked|used|built|"
            r"implemented|developed|applied|created|evaluated|deployed|analysed|analyzed|designed|"
            r"maintained|automated|trained|studied|coursework|certification|certified|internship|"
            r"project|projects|hands[- ]on|practical|demonstrated|programmed|learned|learnt|mastery)\b",
            re.I,
        )
        contrast = re.compile(r"\b(?:but|however|although|whereas|yet)\b", re.I)
        list_separator = re.compile(r",|;|\band\b|/|\|")

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
                    if negative.search(context) or aspirational.search(context):
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
        results: List[CompetencyRequirement] = []
        matched: List[str] = []
        missing: List[str] = []
        learning: List[str] = []
        matched_aliases = set()
        weights = {"required": 1.0, "preferred": self.config.preferred_requirement_weight}
        total_weight = 0.0
        covered_weight = 0.0
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
            total_weight += weights[competency.priority]
            aliases = self._competency_aliases(competency.name)
            present = self._student_mentions_competency(
                competency.name, student_evidence_sentences, known_student_skills
            )
            if present:
                status = "matched"
                covered_weight += weights[competency.priority]
                matched.append(normalize_skill(competency.name))
                matched_aliases.update(aliases)
            else:
                status = "missing"
                normalized_name = normalize_skill(competency.name)
                if competency.priority == "required":
                    missing.append(normalized_name)
                learning.append(normalized_name)
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
        if (
            unresolved
            and semantic_fallback
            and st.backend == "sentence-transformers"
            and student_evidence_vectors is not None
            and len(student_evidence_vectors) > 0
        ):
            prompts = [
                f"The candidate has practical knowledge or experience with {item.name}."
                for _, item in unresolved
            ]
            query_vectors = self._embed(st, prompts)
            similarities = cosine_similarity(query_vectors, student_evidence_vectors)
            for row, (result_index, competency) in enumerate(unresolved):
                if float(similarities[row].max()) < COMPETENCY_SEMANTIC_MATCH_THRESHOLD:
                    continue
                previous = results[result_index]
                results[result_index] = previous.model_copy(update={"status": "matched"})
                normalized_name = normalize_skill(competency.name)
                if normalized_name in missing:
                    missing.remove(normalized_name)
                if normalized_name in learning:
                    learning.remove(normalized_name)
                if normalized_name not in matched:
                    matched.append(normalized_name)
                covered_weight += weights[competency.priority]
                matched_aliases.update(self._competency_aliases(competency.name))

        results.sort(key=lambda item: (item.priority == "not_required", item.name.casefold()))
        skill_percent = (covered_weight / total_weight * 100.0) if total_weight else 0.0
        return _CompetencyAssessment(
            requirements=results,
            matched=sorted(set(matched), key=str.casefold),
            missing=sorted(set(missing), key=str.casefold),
            learning=sorted(set(learning), key=str.casefold),
            skill_match_percent=skill_percent,
            matched_aliases=matched_aliases,
            evaluated_count=sum(item.priority == "required" for item in competencies),
        )

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
        skills = extract_skills(text, st.skill_extractor)
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
        if n_requirements == 0:
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
            recommended_learning=self._order_learning(st, assessment.learning),
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
            recommended_learning=self._order_learning(st, competency_assessment.learning),
            score_breakdown=(
                self._breakdown(
                    skill_pct, sem_pct, category_pct, n_requirements, base_score, academic_penalty
                ) if details else None
            ),
        )
        return rec, skill_pct

    def recommend(self, student_text: str, top_k: int = 3, include_details: bool = False) -> RecommendResponse:
        t0 = time.perf_counter()
        self._check_text(student_text, "student_text")
        k = self._top_k(top_k, 3, 10)
        st = self._ensure()

        profile = self._profile_from_fields(self._extract_fields(st, student_text))
        student_skills = set(profile.skills)
        evidence_sentences = self._student_evidence_sentences(student_text, profile.academic_records)
        semantic_profile = f"{clean_text(student_text)} {profile.profile_summary}"
        sem = self._cos_percent(self._embed(st, [semantic_profile]), st.index_vectors)
        competency_assessments = [
            self._assess_competencies(
                st,
                st.job_descriptions[i],
                list(student_skills),
                evidence_sentences,
                baseline_skills=sorted(st.job_skill_sets[i]),
                competency_candidates=st.job_competencies[i],
                semantic_fallback=False,
            )
            for i in range(len(st.job_ids))
        ]
        skill = np.array(
            [assessment.skill_match_percent for assessment in competency_assessments],
            dtype=np.float64,
        )
        category_proba = st.classifier.predict_proba(
            st.classifier_vectorizer.transform([clean_text(student_text)])
        )[0]
        category_lookup = dict(zip(st.classifier.classes_, category_proba))
        category = np.array(
            [category_lookup.get(job_category, 0.0) * 100.0 for job_category in st.job_categories]
        )
        base_scores = np.round(
            self.config.skill_weight * skill
            + self.config.semantic_weight * sem
            + self.config.category_weight * category,
            2,
        )
        academic_penalties = np.zeros(len(st.job_ids), dtype=np.float64)
        if profile.academic_records:
            for i, (title, description) in enumerate(zip(st.job_titles, st.job_descriptions)):
                adjustments = self._academic_adjustments(
                    st,
                    profile.academic_records,
                    description,
                    demonstrated_competencies=competency_assessments[i].matched_aliases,
                )
                academic_penalties[i] = min(100.0, sum(a.penalty_percent for a in adjustments))
        scores = np.maximum(0.0, base_scores - academic_penalties)
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
            rec, skill_pct = self._job_record(
                st, i, rank, float(sem[i]), float(category[i]),
                profile.academic_records, include_details,
                competency_assessment=competency_assessments[i],
            )
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
            },
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
