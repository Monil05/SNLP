"""Week 3 core NLP pipeline for Group 8 Placement & Career Intelligence.

This is the primary (Week 3) matching pipeline. It is deliberately modular so
it can be wrapped by a production API (see ``api.py`` in this same folder).

It includes:
- domain NER / information extraction for skills and profile fields
- a genuinely derived student profile representation (education + skills +
  experience), combined with sanitised profile context for embedding
- job-category classification (auxiliary component)
- Sentence-Transformer semantic embeddings (the Week 3 embedding model)
- the Week 3 core Match Score: 63% skill-match percentage + 27% semantic
  similarity percentage + 10% category compatibility, on a shared 0-100 scale
- local semantic retrieval ("RAG" retrieval stage) over the FULL cleaned job
  dataset
- extractive summarization (generic text utility)
- the Week 2 TF-IDF + cosine baseline, kept available separately so it can be
  compared against the Week 3 core match (it is NOT used to compute the
  Week 3 Match Score)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# `baseline.py` (the Week 2 baseline) lives one level up, in `src/`, outside
# the Week 3 folder, so it stays clearly separated from Week 3 code.
WEEK3_DIR = Path(__file__).resolve().parent
SRC_DIR = WEEK3_DIR.parent
BASE_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from week2.baseline import (  # noqa: E402
    build_job_text,
    build_skill_vocabulary,
    clean_description,
    extract_cv_text,
    extract_skills,
    fit_tfidf,
    format_skills,
    parse_saved_skills,
    score_student_against_job,
)


DATA_DIR = BASE_DIR / "data"
CLEANED_JOBS_PATH = DATA_DIR / "cleaned_jobs.csv"
TRAIN_PATH = DATA_DIR / "train_jobs.csv"
TEST_PATH = DATA_DIR / "test_jobs.csv"
# Cached Sentence-Transformer embeddings for every job in the full cleaned,
# searchable dataset (see "Dataset / embedding preparation" requirements).
JOB_EMBEDDINGS_CACHE = DATA_DIR / "week3_job_embeddings.npz"
# Local on-disk cache for the Sentence-Transformer model. The first run
# downloads the model weights into this folder; later runs reuse the local
# copy instead of downloading the model again.
SENTENCE_TRANSFORMER_CACHE_DIR = DATA_DIR / "sentence_transformer_model"


# Common degree and profile patterns. This is intentionally lightweight and
# explainable for Week 3; a learned NER model is a stretch goal, not required.
DEGREE_PATTERNS = [
    r"\bB\.?Tech\.?\b",
    r"\bB\.?E\.?\b",
    r"\bB\.?Sc\.?\b",
    r"\bB\.?Com\.?\b",
    r"\bB\.?B\.?A\.?\b",
    r"\bM\.?Tech\.?\b",
    r"\bM\.?E\.?\b",
    r"\bM\.?Sc\.?\b",
    r"\bM\.?Com\.?\b",
    r"\bM\.?B\.?A\.?\b",
    r"\bM\.?S\.?\b",
    r"\bCFA\b",
    r"\bCPA\b",
]

DEGREE_DISPLAY = {
    "btech": "B.Tech",
    "be": "B.E.",
    "bsc": "B.Sc.",
    "bcom": "B.Com.",
    "bba": "BBA",
    "mtech": "M.Tech",
    "me": "M.E.",
    "msc": "M.Sc.",
    "mcom": "M.Com.",
    "mba": "MBA",
    "ms": "M.S.",
    "cfa": "CFA",
    "cpa": "CPA",
}

# Words that stop a specialization phrase from growing further, so that e.g.
# "B.Tech with 3 years experience" is not misread as specializing in "With".
SPECIALIZATION_STOPWORDS = {
    "with", "and", "having", "experience", "years", "year", "student",
    "graduate", "background", "from", "at", "the", "a", "an", "who",
    "currently", "seeking", "looking", "skilled", "proficient", "expert",
}

_SPECIALIZATION_RE = re.compile(r"^\s*[,:\-]?\s*(?:in|of)?\s*((?:[A-Za-z][A-Za-z&]*\s*){1,4})")

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
YEARS_RE = re.compile(r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*\+?\s*(?:years?|yrs?)\b", re.I)


def _canonical_degree_label(token: str) -> str:
    key = re.sub(r"[^a-zA-Z]", "", token).lower()
    return DEGREE_DISPLAY.get(key, token.upper())


def _extract_degree_specialization(remainder: str) -> str:
    """Capture the specialization immediately following a degree mention.

    Handles 'B.Tech in Data Science', 'B.Tech Data Science' and
    'B.Tech, Data Science' without swallowing unrelated trailing text such as
    '... with 3 years experience'.
    """
    match = _SPECIALIZATION_RE.match(remainder)
    if not match:
        return ""
    kept: List[str] = []
    for word in match.group(1).split():
        if word.lower() in SPECIALIZATION_STOPWORDS:
            break
        kept.append(word)
    return " ".join(kept)


def _extract_degrees(text: str) -> List[str]:
    """Extract degree/certification mentions, preserving any specialization
    that immediately follows instead of truncating it to just the degree
    acronym (e.g. 'B.Tech Data Science' -> 'B.Tech in Data Science', not
    just 'B.Tech').
    """
    found: Dict[str, str] = {}
    for pattern in DEGREE_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.I):
            label = _canonical_degree_label(m.group(0))
            specialization = _extract_degree_specialization(text[m.end():])
            normalized = f"{label} in {specialization}" if specialization else label
            key = normalized.lower()
            if key not in found:
                found[key] = normalized
    return list(found.values())


@dataclass
class V1Config:
    # Week 3 requires a proper embedding-based semantic representation, so the
    # Sentence-Transformer backend is the default (not an optional extra).
    use_sentence_transformers: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    summary_sentences: int = 3
    skill_weight: float = 0.63
    semantic_weight: float = 0.27
    category_weight: float = 0.10


class V1NLP:
    """Reusable in-memory NLP service for the Week 3 pipeline."""

    def __init__(
        self,
        train_jobs: pd.DataFrame | None = None,
        test_jobs: pd.DataFrame | None = None,
        jobs: pd.DataFrame | None = None,
        config: V1Config | None = None,
    ):
        self.config = config or V1Config()
        if abs(
            self.config.skill_weight
            + self.config.semantic_weight
            + self.config.category_weight
            - 1.0
        ) > 1e-9:
            raise ValueError("Week 3 score weights must sum to 1.0")

        # Week 3 searches the FULL cleaned job dataset. It must not be
        # restricted to the train split.
        self.jobs = jobs.copy() if jobs is not None else pd.read_csv(CLEANED_JOBS_PATH)
        # Train/validation/test splits stay available for evaluation-style
        # use elsewhere; they are not used as the searchable job index.
        self.train_jobs = train_jobs.copy() if train_jobs is not None else pd.read_csv(TRAIN_PATH)
        self.test_jobs = test_jobs.copy() if test_jobs is not None else pd.read_csv(TEST_PATH)

        # Skill vocabulary is built from the full searchable dataset so a
        # skill that only appears in held-out jobs is never missed.
        self.skill_vocabulary = build_skill_vocabulary(self.jobs)

        # TF-IDF retained for (a) the Week 2 baseline representation, kept
        # available separately for later comparison and fit on the training
        # split exactly like the Week 2 baseline does, and (b) a fallback
        # text representation for the generic embed()/similarity() utilities.
        self.vectorizer = fit_tfidf(self.train_jobs)

        # Fit the category classifier only on the labelled training split.
        # In particular, do not include category names in their own feature
        # text: that would leak the target label into the classifier.
        self.category_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        category_texts = self.jobs.apply(build_job_text, axis=1)
        classifier_texts = (
            self.train_jobs["job_title"].map(clean_description).str.lower()
            + " "
            + self.train_jobs["job_description"].map(clean_description).str.lower()
        )
        self.category_matrix = self.category_vectorizer.fit_transform(classifier_texts)
        self.category_classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.category_classifier.fit(self.category_matrix, self.train_jobs["category"].astype(str))

        # Fields needed for the Week 3 searchable job index (the full,
        # cleaned dataset — see "Dataset / embedding preparation").
        self._job_ids = self.jobs["job_id"].astype(str).tolist()
        self._job_titles = self.jobs["job_title"].astype(str).tolist()
        self._job_categories = self.jobs["category"].astype(str).tolist()
        self._job_descriptions = self.jobs["job_description"].astype(str).tolist()
        self._job_required_skills = [set(parse_saved_skills(value)) for value in self.jobs["job_skill_set"]]
        self._job_texts = category_texts.tolist()

        self._semantic_model = None
        if self.config.use_sentence_transformers:
            self._load_semantic_model()

        # Generate (or load previously stored) embeddings for every job in
        # the full cleaned searchable dataset.
        self._job_index_vectors = self._build_job_index()

    # ------------------------------------------------------------------
    # Embedding backend
    # ------------------------------------------------------------------
    def _load_semantic_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The Week 3/Week 4 improved model requires the "
                "'sentence-transformers' package. Install the project "
                "requirements with:\n"
                f"  {sys.executable} -m pip install -r \"{BASE_DIR / 'requirements.txt'}\"\n"
                "Then run the project again. This is a dependency/setup "
                "issue, not a model-code error."
            ) from exc
        # Keep the Sentence-Transformer model inside the project data folder.
        # Prefer an explicit local snapshot when available. Passing the Hub
        # name even with ``cache_folder`` can still trigger slow network/HF
        # resolution before the local cache is consulted.
        SENTENCE_TRANSFORMER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        repo = (
            self.config.embedding_model
            if "/" in self.config.embedding_model
            else f"sentence-transformers/{self.config.embedding_model}"
        )
        snapshots = SENTENCE_TRANSFORMER_CACHE_DIR / (
            "models--" + repo.replace("/", "--")
        ) / "snapshots"
        local_snapshot = next(
            (
                path for path in sorted(snapshots.iterdir())
                if (path / "config.json").exists()
                and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())
            ),
            None,
        ) if snapshots.exists() else None
        try:
            self._semantic_model = (
                SentenceTransformer(str(local_snapshot))
                if local_snapshot is not None
                else SentenceTransformer(
                    self.config.embedding_model,
                    cache_folder=str(SENTENCE_TRANSFORMER_CACHE_DIR),
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Sentence-Transformer model '{self.config.embedding_model}'. "
                "On the first run, an internet connection is required so the model "
                "weights can be downloaded. After that, the cached model is reused "
                f"from: {SENTENCE_TRANSFORMER_CACHE_DIR}"
            ) from exc

    def _embed_texts(self, texts: Sequence[str]):
        cleaned = [clean_description(t) for t in texts]
        if self._semantic_model is not None:
            return np.asarray(self._semantic_model.encode(cleaned, normalize_embeddings=True))
        return self.vectorizer.transform(cleaned)

    def _cosine_percent(self, vector_a, vector_b) -> float:
        score = float(cosine_similarity(vector_a, vector_b)[0][0])
        return float(np.clip(score * 100.0, 0.0, 100.0))

    def _build_job_index(self):
        """Generate and store embeddings for all jobs in the full cleaned
        searchable dataset, reusing a cached copy when it is still valid."""
        if self._semantic_model is not None:
            cached = self._load_job_embedding_cache()
            if cached is not None:
                return cached
            vectors = self._semantic_model.encode(
                [clean_description(text) for text in self._job_texts],
                normalize_embeddings=True,
                batch_size=128,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            self._save_job_embedding_cache(vectors)
            return vectors
        return self.vectorizer.transform([clean_description(text) for text in self._job_texts])

    def _load_job_embedding_cache(self):
        if not JOB_EMBEDDINGS_CACHE.exists():
            return None
        try:
            cached = np.load(JOB_EMBEDDINGS_CACHE, allow_pickle=False)
            same_model = cached["model"].item() == self.config.embedding_model
            same_ids = list(cached["job_ids"]) == self._job_ids
            if same_model and same_ids:
                return cached["embeddings"]
        except Exception:
            return None
        return None

    def _save_job_embedding_cache(self, vectors: np.ndarray) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                JOB_EMBEDDINGS_CACHE,
                embeddings=vectors.astype(np.float32),
                job_ids=np.array(self._job_ids),
                model=np.array(self.config.embedding_model),
            )
        except OSError:
            pass  # Caching is a best-effort optimization; ignore write failures.

    # ------------------------------------------------------------------
    # Information extraction
    # ------------------------------------------------------------------
    def extract_entities(self, text: str) -> Dict[str, Any]:
        text = clean_description(text)
        skills = extract_skills(text, self.skill_vocabulary)
        degrees = _extract_degrees(text)
        emails = list(dict.fromkeys(EMAIL_RE.findall(text)))
        phones = list(dict.fromkeys(PHONE_RE.findall(text)))
        years = list(dict.fromkeys(m.group(0) for m in YEARS_RE.finditer(text)))
        return {
            "skills": skills,
            "degrees_or_certifications": degrees,
            "emails": emails,
            "phone_numbers": phones,
            "experience_mentions": years,
            "entities_count": len(skills) + len(degrees) + len(emails) + len(phones) + len(years),
        }

    def derive_student_profile(self, text: str) -> Dict[str, Any]:
        """Build a genuinely derived student representation (normalized
        education + extracted skills + available experience information)
        instead of just reformatting the raw input text. This representation
        also doubles as the embedding input for semantic similarity."""
        entities = self.extract_entities(text)
        education = "; ".join(entities["degrees_or_certifications"]) or "Not specified"
        skills = entities["skills"]
        skills_text = "; ".join(skills) if skills else "Not specified"
        experience = "; ".join(entities["experience_mentions"]) or "Not specified"
        profile_summary = f"Education: {education}. Skills: {skills_text}. Experience: {experience}."
        return {
            "normalized_education": education,
            "skills": skills,
            "experience": experience,
            "profile_summary": profile_summary,
            # Keep useful professional context (for example, "Cybersecurity
            # Student") that exact extraction does not model, while removing
            # contact details before the text reaches the embedding model.
            "semantic_text": clean_description(
                PHONE_RE.sub(" ", EMAIL_RE.sub(" ", text))
                + " "
                + profile_summary
            ).strip(),
        }

    # ------------------------------------------------------------------
    # Classification (auxiliary component)
    # ------------------------------------------------------------------
    def classify_job(self, text: str) -> Dict[str, Any]:
        vector = self.category_vectorizer.transform([clean_description(text)])
        probabilities = self.category_classifier.predict_proba(vector)[0]
        order = np.argsort(probabilities)[::-1]
        top = [
            {"category": str(self.category_classifier.classes_[i]), "confidence": round(float(probabilities[i]), 4)}
            for i in order[: min(5, len(order))]
        ]
        return {"predicted_category": top[0]["category"], "confidence": top[0]["confidence"], "top_categories": top}

    # ------------------------------------------------------------------
    # Generic embedding / similarity / summarization utilities
    # ------------------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> Dict[str, Any]:
        vectors = self._embed_texts(texts)
        if self._semantic_model is not None:
            return {"model": self.config.embedding_model, "dimensions": int(vectors.shape[1]), "vectors": vectors.tolist()}
        return {"model": "tfidf-v1", "dimensions": int(vectors.shape[1]), "vectors": vectors.toarray().tolist()}

    def similarity(self, text_a: str, text_b: str) -> Dict[str, Any]:
        vectors = self._embed_texts([text_a, text_b])
        score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        model = self.config.embedding_model if self._semantic_model is not None else "tfidf-v1"
        return {"similarity": round(score, 4), "similarity_percent": round(score * 100, 2), "model": model}

    def summarize(self, text: str, max_sentences: int | None = None) -> Dict[str, Any]:
        text = clean_description(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        n = max(1, max_sentences or self.config.summary_sentences)
        if len(sentences) <= n:
            return {"summary": " ".join(sentences), "sentences_used": len(sentences)}
        matrix = self.vectorizer.transform(sentences)
        centroid = np.asarray(matrix.mean(axis=0))
        scores = cosine_similarity(matrix, centroid).ravel()
        chosen = sorted(np.argsort(scores)[-n:].tolist())
        summary = " ".join(sentences[i] for i in chosen)
        return {"summary": summary, "sentences_used": n}

    # ------------------------------------------------------------------
    # Retrieval over the full cleaned job dataset
    # ------------------------------------------------------------------
    def retrieve_jobs(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        k = max(1, min(top_k or self.config.top_k, len(self.jobs)))
        query_vector = self._embed_texts([query])
        scores = cosine_similarity(query_vector, self._job_index_vectors).ravel()
        indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in indices:
            idx = int(idx)
            results.append({
                "job_id": self._job_ids[idx],
                "category": self._job_categories[idx],
                "job_title": self._job_titles[idx],
                "similarity": round(float(scores[idx]), 4),
                "job_description": self._job_descriptions[idx],
                "skills": format_skills(sorted(self._job_required_skills[idx])),
            })
        return results

    def rag(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        retrieved = self.retrieve_jobs(query, top_k)
        context = "\n\n".join(
            f"Job {r['job_id']} | {r['job_title']} | {r['category']}\n{r['job_description']}"
            for r in retrieved
        )
        return {
            "query": query,
            "retrieved_jobs": retrieved,
            "context": context,
            "note": "Week 3 retrieval stage over the full cleaned job dataset; a generative answer layer can be added later without changing retrieval.",
        }

    # ------------------------------------------------------------------
    # Week 3 core matching: skill (63%) + semantic (27%) + category (10%)
    # ------------------------------------------------------------------
    def _skill_match_percent(self, matched: Sequence[str], required: Sequence[str]) -> float:
        return (len(matched) / len(required)) * 100.0 if required else 0.0

    def _category_probability_percent(self, student_text: str, category: str) -> float:
        """How compatible a student's profile is with a job category.

        The classifier is trained only on the training split.  This is a small
        reranking signal, not a replacement for skill evidence or semantic
        similarity.
        """
        vector = self.category_vectorizer.transform([clean_description(student_text)])
        probabilities = self.category_classifier.predict_proba(vector)[0]
        probabilities_by_category = dict(zip(self.category_classifier.classes_, probabilities))
        return float(probabilities_by_category.get(category, 0.0) * 100.0)

    def _combine_score(
        self,
        skill_match_percent: float,
        semantic_similarity_percent: float,
        category_compatibility_percent: float,
    ) -> float:
        return (
            self.config.skill_weight * skill_match_percent
            + self.config.semantic_weight * semantic_similarity_percent
            + self.config.category_weight * category_compatibility_percent
        )

    def score_against_job_text(self, student_text: str, job_text: str) -> Dict[str, Any]:
        """Full Week 3 scoring detail for an ad-hoc job description (the JD
        PROVIDED flow). Kept available internally for debugging/evaluation;
        see match_with_jd() for the field-restricted, user-facing output."""
        profile = self.derive_student_profile(student_text)
        student_skills = set(profile["skills"])
        job_skills = set(extract_skills(job_text, self.skill_vocabulary))

        matched = sorted(student_skills & job_skills)
        missing = sorted(job_skills - student_skills)
        skill_match_percent = self._skill_match_percent(matched, job_skills)

        student_vector = self._embed_texts([profile["semantic_text"]])
        job_vector = self._embed_texts([clean_description(job_text)])
        semantic_similarity_percent = self._cosine_percent(student_vector, job_vector)
        job_category = self.classify_job(job_text)["predicted_category"]
        category_compatibility_percent = self._category_probability_percent(student_text, job_category)

        match_score = round(
            self._combine_score(
                skill_match_percent,
                semantic_similarity_percent,
                category_compatibility_percent,
            ),
            2,
        )

        return {
            "match_score": match_score,
            # Kept separate from the final Match Score, for internal use only.
            "skill_match_percent": round(skill_match_percent, 2),
            "semantic_similarity_percent": round(semantic_similarity_percent, 2),
            "category_compatibility_percent": round(category_compatibility_percent, 2),
            "matched_skills": format_skills(matched),
            "missing_weak_skills": format_skills(missing),
            "recommended_learning": format_skills(missing),
            "student_profile": profile,
        }

    def match_with_jd(self, student_text: str, job_text: str) -> Dict[str, Any]:
        """JD PROVIDED flow. Returns ONLY the fields the API is allowed to
        expose: Match Score, Matched Skills, Missing/Weak Skills and
        Recommended Learning."""
        detail = self.score_against_job_text(student_text, job_text)
        return {
            "match_score": detail["match_score"],
            "matched_skills": detail["matched_skills"],
            "missing_weak_skills": detail["missing_weak_skills"],
            "recommended_learning": detail["recommended_learning"],
        }

    def rank_jobs_for_student(self, student_text: str) -> List[Dict[str, Any]]:
        """Full Week 3 ranking detail across the full cleaned job dataset
        (the JD NOT PROVIDED flow). Kept available internally for
        debugging/evaluation; see recommend_job() for the field-restricted,
        user-facing output."""
        profile = self.derive_student_profile(student_text)
        student_skills = set(profile["skills"])

        student_vector = self._embed_texts([profile["semantic_text"]])
        semantic_scores = cosine_similarity(student_vector, self._job_index_vectors).ravel()
        semantic_percents = np.clip(semantic_scores * 100.0, 0.0, 100.0)
        category_vector = self.category_vectorizer.transform([clean_description(student_text)])
        category_probabilities = dict(
            zip(self.category_classifier.classes_, self.category_classifier.predict_proba(category_vector)[0])
        )

        results = []
        for idx, required in enumerate(self._job_required_skills):
            matched = sorted(student_skills & required)
            missing = sorted(required - student_skills)
            skill_match_percent = self._skill_match_percent(matched, required)
            semantic_similarity_percent = float(semantic_percents[idx])
            category_compatibility_percent = float(
                category_probabilities.get(self._job_categories[idx], 0.0) * 100.0
            )
            match_score = round(
                self._combine_score(
                    skill_match_percent,
                    semantic_similarity_percent,
                    category_compatibility_percent,
                ),
                2,
            )
            results.append({
                "job_id": self._job_ids[idx],
                "job_title": self._job_titles[idx],
                "category": self._job_categories[idx],
                "match_score": match_score,
                "skill_match_percent": round(skill_match_percent, 2),
                "semantic_similarity_percent": round(semantic_similarity_percent, 2),
                "category_compatibility_percent": round(category_compatibility_percent, 2),
                "matched_skills": format_skills(matched),
                "missing_weak_skills": format_skills(missing),
                "recommended_learning": format_skills(missing),
            })
        results.sort(key=lambda item: item["match_score"], reverse=True)
        return results

    def _clean_recommended_job_title(self, title: str) -> str:
        """Return the simplified title shown to the user.

        Dataset job titles can contain level numbers, parenthetical
        specializations and internal requisition codes, for example:
        ``data solutions analyst ii (managed care - finance focus) - r7269``.
        The recommendation UI should expose only the actual job title.
        """
        cleaned = clean_description(title)

        # Remove parenthetical specialization/focus text.
        cleaned = re.sub(r"\s*\([^)]*\)", "", cleaned)

        # Remove internal requisition/reference codes such as "- r7269".
        cleaned = re.sub(r"\s*-\s*r\d+\s*$", "", cleaned, flags=re.I)

        # Remove trailing Roman-numeral job levels such as I, II, III, IV.
        cleaned = re.sub(
            r"\s+(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\s*$",
            "",
            cleaned,
            flags=re.I,
        )

        return re.sub(r"\s+", " ", cleaned).strip()

    def recommend_job(self, student_text: str) -> Dict[str, Any]:
        """JD NOT PROVIDED flow. Searches/ranks the FULL cleaned job dataset
        and returns only user-facing recommendation fields. Internal job IDs,
        categories and dataset-specific title suffixes are not exposed.
        """
        ranked = self.rank_jobs_for_student(student_text)
        top = ranked[0]
        return {
            "recommended_job": self._clean_recommended_job_title(top["job_title"]),
            "match_score": top["match_score"],
            "matched_skills": top["matched_skills"],
            "missing_weak_skills": top["missing_weak_skills"],
            "recommended_learning": top["recommended_learning"],
        }

    # ------------------------------------------------------------------
    # Week 2 baseline (kept available separately, for comparison only)
    # ------------------------------------------------------------------
    def _baseline_match(self, student_text: str, job_text: str) -> Dict[str, Any]:
        """Week 2 TF-IDF + skill-coverage baseline, kept available only for
        side-by-side comparison with the Week 3 core match (see pipeline()).
        This is NOT used to compute the Week 3 Match Score."""
        return score_student_against_job(
            student_text,
            {
                "category": "user_input",
                "job_title": "user_input",
                "job_description": job_text,
                "job_skill_set": extract_skills(job_text, self.skill_vocabulary),
            },
            self.vectorizer,
            self.skill_vocabulary,
        )

    # ------------------------------------------------------------------
    # CV processing / internal debug pipeline
    # ------------------------------------------------------------------
    def process_cv(self, path: str | Path) -> Dict[str, Any]:
        path = Path(path)
        text = extract_cv_text(path)
        if not text.strip():
            raise ValueError(f"No text could be extracted from CV: {path.name}")
        return {
            "filename": path.name,
            "text": text,
            "entities": self.extract_entities(text),
            "summary": self.derive_student_profile(text),
        }

    def pipeline(self, student_text: str, job_text: str, top_k: int | None = None) -> Dict[str, Any]:
        """Internal debug/demo view combining every component, including the
        Week 2 baseline kept for comparison. This is NOT one of the two
        required user-facing API flows — see match_with_jd() and
        recommend_job() for those."""
        entities = self.extract_entities(student_text)
        student_profile = self.derive_student_profile(student_text)
        classification = self.classify_job(job_text)
        week3_match = self.score_against_job_text(student_text, job_text)
        baseline_match = self._baseline_match(student_text, job_text)
        retrieval = self.retrieve_jobs(job_text, top_k)
        return {
            "version": "1.0",
            "student": {"entities": entities, "summary": student_profile},
            "job": {"classification": classification, "entities": self.extract_entities(job_text)},
            "match": week3_match,
            "baseline_match": baseline_match,
            "rag_retrieval": retrieval,
        }
