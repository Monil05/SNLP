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

    def with_overrides(self, **changes) -> "AssetConfig":
        return replace(self, **changes)

    @classmethod
    def from_env(cls) -> "AssetConfig":
        """Build a config from ``CAREER_NLP_*`` environment variables.

        CAREER_NLP_JOBS_PATH, CAREER_NLP_CACHE_DIR, CAREER_NLP_MODEL_CACHE_DIR,
        CAREER_NLP_BACKEND, CAREER_NLP_EMBEDDING_MODEL, CAREER_NLP_WARMUP
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
        kwargs["warmup_on_start"] = _env_bool("CAREER_NLP_WARMUP", base.warmup_on_start)
        return replace(base, **kwargs)
