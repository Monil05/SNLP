"""Exception hierarchy. Each error carries a stable machine-readable ``code``
and an HTTP status so the REST layer can map it without string matching."""

from __future__ import annotations

from typing import Any, Dict, Optional


class CareerNLPError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InputValidationError(CareerNLPError):
    code = "invalid_input"
    http_status = 422


class CVProcessingError(CareerNLPError):
    code = "cv_processing_failed"
    http_status = 422


class ModelLoadError(CareerNLPError):
    code = "model_load_failed"
    http_status = 503


class NotReadyError(CareerNLPError):
    code = "not_ready"
    http_status = 503
