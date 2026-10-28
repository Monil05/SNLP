"""career_nlp — reusable Placement & Career Intelligence NLP asset.

Quick start::

    from career_nlp import CareerNLP

    nlp = CareerNLP()                       # lazy: nothing heavy happens yet
    result = nlp.match(student_text, job_text)   # JD provided
    print(result.match_score, result.confidence.level)

    result = nlp.recommend(student_text)         # JD not provided
    print(result.recommended_job.job_title)

Every public method returns a validated Pydantic model (see ``schemas``), so the
same objects can be used in Python, serialised to JSON, or served over REST via
``career_nlp.api.create_app``.
"""

from .config import AssetConfig
from .engine import CareerNLP
from .errors import (
    CareerNLPError,
    CVProcessingError,
    InputValidationError,
    ModelLoadError,
    NotReadyError,
)
from .version import ASSET_VERSION, SCHEMA_VERSION

__version__ = ASSET_VERSION

__all__ = [
    "AssetConfig",
    "CareerNLP",
    "CareerNLPError",
    "CVProcessingError",
    "InputValidationError",
    "ModelLoadError",
    "NotReadyError",
    "ASSET_VERSION",
    "SCHEMA_VERSION",
    "__version__",
]
