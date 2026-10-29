"""Runtime configuration for the asset.

Everything that was a hard-coded path or constant in the Week 3 pipeline is a
field here, so the component can be dropped into another project and pointed at
its own job dataset / model cache without editing source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

# <repo>/base version/src/career_nlp/config.py -> <repo>/base version
_PROJECT_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _PROJECT_DIR / "data"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AssetConfig:
    # --- data ---------------------------------------------------------------
    jobs_path: Path = _DEFAULT_DATA_DIR / "cleaned_jobs.csv"
    cache_dir: Path = _DEFAULT_DATA_DIR / "career_nlp_cache"
    model_cache_dir: Path = _DEFAULT_DATA_DIR / "sentence_transformer_model"

    # --- embedding backend --------------------------------------------------
    # "sentence-transformers": the Week 3 core backend (all-MiniLM-L6-v2).
    # "lsa": dense TF-IDF + truncated SVD. Offline/CI fallback ONLY - it is not
    #        the Week 3 representation and scores will differ.
    # "auto": sentence-transformers if importable, else lsa (with a warning).
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    lsa_dimensions: int = 256

    # --- scoring ------------------------------------------------------------
    # The core ranker is a hybrid: exact skill evidence, dense semantic
    # similarity, and a small classifier-based category compatibility signal.
    skill_weight: float = 0.63
    semantic_weight: float = 0.27
    category_weight: float = 0.10
    poor_grade_threshold_percent: float = 50.0
    academic_subject_penalty: float = 40.0
    academic_core_domain_penalty: float = 60.0
    # A fully covered preferred tier is worth this much relative to a fully
    # covered required tier. Individual preferred skills share that tier, so a
    # long preferred list cannot outweigh a few essential requirements.
    preferred_requirement_weight: float = 0.35
    # Soft and contextual competencies share a smaller tier. An explicit
    # "required"/"must" statement still promotes one of them into the required tier.
    generic_requirement_weight: float = 0.12
    # Student-only recommend() scores this many retrieved jobs in detail.
    candidate_shortlist_size: int = 180
    # Slots reserved for jobs that share skills with the student, including
    # generic skills when those are the profile's only evidence.
    retrieval_skill_slots: int = 100
    # Upper bound on paraphrase checks embedded for one recommendation request.
    semantic_fallback_max_prompts: int = 48
    generic_competencies: frozenset = field(default_factory=lambda: frozenset({
        "communication",
        "collaboration",
        "adaptability",
        "customer focus",
        "customer service",
        "problem solving",
        "innovation",
        "teamwork",
        "team work",
        "leadership",
        "interpersonal",
        "interpersonal skills",
        "time management",
        "attention to detail",
        "critical thinking",
        "flexibility",
        "creativity",
        "work ethic",
        "presentation skills",
        "organizational skills",
    }))

    # --- limits / behaviour -------------------------------------------------
    default_top_k: int = 5
    max_top_k: int = 20
    summary_sentences: int = 3
    max_text_chars: int = 50_000
    max_cv_bytes: int = 5 * 1024 * 1024
    warmup_on_start: bool = True

    def __post_init__(self):
        if self.embedding_backend not in {"sentence-transformers", "lsa", "auto"}:
            raise ValueError(
                "embedding_backend must be 'sentence-transformers', 'lsa' or 'auto'"
            )
        if abs((self.skill_weight + self.semantic_weight + self.category_weight) - 1.0) > 1e-9:
            raise ValueError("skill_weight + semantic_weight + category_weight must equal 1.0")
        if self.max_text_chars < 1 or self.default_top_k < 1:
            raise ValueError("max_text_chars and default_top_k must be >= 1")
        if not 0.0 <= self.poor_grade_threshold_percent <= 100.0:
            raise ValueError("poor_grade_threshold_percent must be between 0 and 100")
        if not 0.0 <= self.academic_subject_penalty <= 100.0:
            raise ValueError("academic_subject_penalty must be between 0 and 100")
        if not 0.0 <= self.academic_core_domain_penalty <= 100.0:
            raise ValueError("academic_core_domain_penalty must be between 0 and 100")
        if not 0.0 <= self.preferred_requirement_weight <= 1.0:
            raise ValueError("preferred_requirement_weight must be between 0 and 1")
        if not 0.0 <= self.generic_requirement_weight <= 1.0:
            raise ValueError("generic_requirement_weight must be between 0 and 1")
        if self.generic_requirement_weight > self.preferred_requirement_weight:
            raise ValueError("generic_requirement_weight cannot exceed preferred_requirement_weight")
        if self.candidate_shortlist_size < 1 or self.retrieval_skill_slots < 1:
            raise ValueError("candidate_shortlist_size and retrieval_skill_slots must be >= 1")
        if self.retrieval_skill_slots > self.candidate_shortlist_size:
            raise ValueError("retrieval_skill_slots cannot exceed candidate_shortlist_size")
        if self.semantic_fallback_max_prompts < 0:
            raise ValueError("semantic_fallback_max_prompts must be >= 0")
        if not isinstance(self.generic_competencies, frozenset):
            object.__setattr__(self, "generic_competencies", frozenset(self.generic_competencies))
        if self.academic_core_domain_penalty < self.academic_subject_penalty:
            # Preserve existing configurations that override only the base
            # penalty. The core-domain adjustment cannot be weaker than it.
            object.__setattr__(self, "academic_core_domain_penalty", self.academic_subject_penalty)

    def with_overrides(self, **changes) -> "AssetConfig":
        return replace(self, **changes)

    @classmethod
    def from_env(cls) -> "AssetConfig":
        """Build a config from ``CAREER_NLP_*`` environment variables.

        CAREER_NLP_JOBS_PATH, CAREER_NLP_CACHE_DIR, CAREER_NLP_MODEL_CACHE_DIR,
        CAREER_NLP_BACKEND, CAREER_NLP_EMBEDDING_MODEL, CAREER_NLP_WARMUP,
        CAREER_NLP_POOR_GRADE_THRESHOLD_PERCENT, CAREER_NLP_ACADEMIC_SUBJECT_PENALTY,
        CAREER_NLP_ACADEMIC_CORE_DOMAIN_PENALTY, CAREER_NLP_PREFERRED_REQUIREMENT_WEIGHT,
        CAREER_NLP_GENERIC_REQUIREMENT_WEIGHT, CAREER_NLP_CANDIDATE_SHORTLIST_SIZE,
        CAREER_NLP_RETRIEVAL_SKILL_SLOTS, CAREER_NLP_SEMANTIC_FALLBACK_MAX_PROMPTS
        """
        base = cls()
        kwargs = {}
        if os.environ.get("CAREER_NLP_JOBS_PATH"):
            kwargs["jobs_path"] = Path(os.environ["CAREER_NLP_JOBS_PATH"])
        if os.environ.get("CAREER_NLP_CACHE_DIR"):
            kwargs["cache_dir"] = Path(os.environ["CAREER_NLP_CACHE_DIR"])
        if os.environ.get("CAREER_NLP_MODEL_CACHE_DIR"):
            kwargs["model_cache_dir"] = Path(os.environ["CAREER_NLP_MODEL_CACHE_DIR"])
        if os.environ.get("CAREER_NLP_BACKEND"):
            kwargs["embedding_backend"] = os.environ["CAREER_NLP_BACKEND"]
        if os.environ.get("CAREER_NLP_EMBEDDING_MODEL"):
            kwargs["embedding_model"] = os.environ["CAREER_NLP_EMBEDDING_MODEL"]
        if os.environ.get("CAREER_NLP_POOR_GRADE_THRESHOLD_PERCENT"):
            kwargs["poor_grade_threshold_percent"] = float(
                os.environ["CAREER_NLP_POOR_GRADE_THRESHOLD_PERCENT"]
            )
        if os.environ.get("CAREER_NLP_ACADEMIC_SUBJECT_PENALTY"):
            kwargs["academic_subject_penalty"] = float(
                os.environ["CAREER_NLP_ACADEMIC_SUBJECT_PENALTY"]
            )
        if os.environ.get("CAREER_NLP_ACADEMIC_CORE_DOMAIN_PENALTY"):
            kwargs["academic_core_domain_penalty"] = float(
                os.environ["CAREER_NLP_ACADEMIC_CORE_DOMAIN_PENALTY"]
            )
        if os.environ.get("CAREER_NLP_PREFERRED_REQUIREMENT_WEIGHT"):
            kwargs["preferred_requirement_weight"] = float(
                os.environ["CAREER_NLP_PREFERRED_REQUIREMENT_WEIGHT"]
            )
        if os.environ.get("CAREER_NLP_GENERIC_REQUIREMENT_WEIGHT"):
            kwargs["generic_requirement_weight"] = float(
                os.environ["CAREER_NLP_GENERIC_REQUIREMENT_WEIGHT"]
            )
        if os.environ.get("CAREER_NLP_CANDIDATE_SHORTLIST_SIZE"):
            kwargs["candidate_shortlist_size"] = int(os.environ["CAREER_NLP_CANDIDATE_SHORTLIST_SIZE"])
        if os.environ.get("CAREER_NLP_RETRIEVAL_SKILL_SLOTS"):
            kwargs["retrieval_skill_slots"] = int(os.environ["CAREER_NLP_RETRIEVAL_SKILL_SLOTS"])
        if os.environ.get("CAREER_NLP_SEMANTIC_FALLBACK_MAX_PROMPTS"):
            kwargs["semantic_fallback_max_prompts"] = int(
                os.environ["CAREER_NLP_SEMANTIC_FALLBACK_MAX_PROMPTS"]
            )
        kwargs["warmup_on_start"] = _env_bool("CAREER_NLP_WARMUP", base.warmup_on_start)
        return replace(base, **kwargs)
