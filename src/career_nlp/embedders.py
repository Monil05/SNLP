"""Pluggable embedding backends.

Any object with ``name``, ``dimensions`` and ``encode(texts) -> (n, d) float32
L2-normalised ndarray`` can be passed to ``CareerNLP(embedder=...)``.

* ``SentenceTransformerEmbedder`` - the Week 3 core backend. Loads from a local
  Hugging Face cache snapshot when one exists (no network), otherwise downloads
  on first use.
* ``LSAEmbedder`` - dense TF-IDF + TruncatedSVD, fitted on the job corpus.
  Offline/CI fallback with no heavy dependencies. NOT equivalent to Week 3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from .errors import ModelLoadError

log = logging.getLogger("career_nlp")


class Embedder(Protocol):
    name: str
    dimensions: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def find_local_snapshot(model_name: str, cache_dir: Path) -> Optional[Path]:
    """Locate an already-downloaded HF snapshot, e.g.
    <cache>/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>/"""
    if Path(model_name).is_dir():
        return Path(model_name)
    if not cache_dir.exists():
        return None
    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    folder = cache_dir / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not folder.exists():
        return None
    for snap in sorted(folder.iterdir()):
        if (snap / "config.json").exists() and (
            (snap / "model.safetensors").exists() or (snap / "pytorch_model.bin").exists()
        ):
            return snap
    return None


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, cache_dir: Path):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelLoadError(
                "The 'sentence-transformers' package is not installed. "
                "Run: pip install sentence-transformers  (or set "
                "embedding_backend='lsa' for the offline fallback).",
                {"package": "sentence-transformers"},
            ) from exc
        self.name = model_name
        snapshot = find_local_snapshot(model_name, cache_dir)
        try:
            if snapshot is not None:
                log.info("Loading %s from local snapshot %s", model_name, snapshot)
                self._model = SentenceTransformer(str(snapshot))
            else:
                cache_dir.mkdir(parents=True, exist_ok=True)
                log.info("Downloading %s (first run) into %s", model_name, cache_dir)
                self._model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        except Exception as exc:  # network / corrupt cache / bad name
            raise ModelLoadError(
                f"Could not load Sentence-Transformer '{model_name}'. The first run "
                f"needs internet access; afterwards the cache at {cache_dir} is reused.",
                {"model": model_name, "cache_dir": str(cache_dir)},
            ) from exc
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class LSAEmbedder:
    """TF-IDF -> TruncatedSVD -> L2 norm. Fit once on the job corpus."""

    def __init__(self, corpus: Sequence[str], dimensions: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name = f"lsa-{dimensions}"
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True)
        matrix = self._tfidf.fit_transform(corpus)
        n_components = max(2, min(dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(matrix)
        self.dimensions = int(n_components)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return _l2_normalise(self._svd.transform(self._tfidf.transform(list(texts))))
