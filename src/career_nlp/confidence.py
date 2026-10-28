"""Confidence scoring.

Two different kinds of confidence live here - the distinction matters:

1. CLASSIFICATION: the classifier's predicted probability. Level thresholds come
   from a held-out reliability check (25% stratified split of the job dataset,
   job-length text): accuracy was 100% at p>=0.60 (n=132), 96-98% for
   0.40<=p<0.60 and ~87% below 0.40. See docs "Confidence scores".

2. MATCH / RECOMMEND: an *evidence-sufficiency* heuristic. It says how much
   signal the score rests on (e.g. a JD in which only one skill was recognised
   gives a skill-match of exactly 0% or 100% - technically a number, but not a
   trustworthy one). It is NOT a calibrated probability that the match is
   correct, and ``Confidence.calibrated`` is False for these.

All weights/thresholds are module constants so they are easy to audit and tune.
"""

from __future__ import annotations

from typing import Dict, Optional

from .schemas import Confidence
from .version import CONFIDENCE_METHOD

# classification (probability thresholds, from reliability table above)
CLASSIFY_HIGH = 0.60
CLASSIFY_MEDIUM = 0.40

# evidence heuristics
HIGH = 0.70
MEDIUM = 0.40
SKILLS_FOR_FULL_EVIDENCE = 5          # >=5 recognised skills = "enough evidence"
MARGIN_FOR_FULL_CONFIDENCE = 15.0     # top1-top2 gap (score points) deemed decisive
# If fewer than half of the top-5 jobs share the top job's category, the
# recommendation is category-ambiguous (Week 5: SALES vs BUSINESS-DEVELOPMENT,
# IT vs generic analytics) and confidence is capped at "low".
CATEGORY_SPLIT_THRESHOLD = 0.5


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _level(score: float, high: float, medium: float) -> str:
    return "high" if score >= high else ("medium" if score >= medium else "low")


def classification_confidence(
    top_probability: float, runner_up: Optional[float], include_factors: bool = True
) -> Confidence:
    score = _clip(top_probability)
    factors: Dict[str, float] = {"top_probability": round(score, 4)}
    if runner_up is not None:
        factors["margin_to_runner_up"] = round(_clip(top_probability - runner_up), 4)
    return Confidence(
        score=round(score, 4),
        level=_level(score, CLASSIFY_HIGH, CLASSIFY_MEDIUM),
        method="classifier_probability",
        calibrated=False,
        factors=factors if include_factors else None,
    )


def match_confidence(
    n_required_skills: int, n_student_skills: int, include_factors: bool = False
) -> Confidence:
    """JD-provided flow. Geometric mean of how much evidence each side offers."""
    jd_evidence = _clip(n_required_skills / SKILLS_FOR_FULL_EVIDENCE)
    profile_evidence = _clip(n_student_skills / SKILLS_FOR_FULL_EVIDENCE)
    score = (jd_evidence * profile_evidence) ** 0.5
    return Confidence(
        score=round(score, 4),
        level=_level(score, HIGH, MEDIUM),
        method=CONFIDENCE_METHOD,
        calibrated=False,
        factors=(
            {"jd_skill_evidence": round(jd_evidence, 4),
             "profile_skill_evidence": round(profile_evidence, 4)}
            if include_factors else None
        ),
    )


def recommend_confidence(
    top_score: float,
    second_score: Optional[float],
    top_skill_match_percent: float,
    category_agreement: float,
    n_student_skills: int,
    include_factors: bool = False,
) -> Confidence:
    """JD-not-provided flow.

    category_agreement = share of the top-5 jobs in the same category as the top
    job; low agreement is exactly the adjacent-category confusion found in the
    Week 5 error analysis (SALES vs BUSINESS-DEVELOPMENT etc.). When it falls
    below CATEGORY_SPLIT_THRESHOLD the score is capped so the level is "low".

    NOTE: that cap was chosen after inspecting the 16 synthetic students, so its
    apparent accuracy on them is in-sample. Re-validate with
    ``examples/validate_confidence.py`` on your own data / backend.
    """
    margin = 1.0 if second_score is None else _clip((top_score - second_score) / MARGIN_FOR_FULL_CONFIDENCE)
    skill_cov = _clip(top_skill_match_percent / 100.0)
    profile_evidence = _clip(n_student_skills / SKILLS_FOR_FULL_EVIDENCE)
    agreement = _clip(category_agreement)
    score = 0.30 * agreement + 0.25 * skill_cov + 0.25 * profile_evidence + 0.20 * margin
    if agreement < CATEGORY_SPLIT_THRESHOLD:
        score = min(score, MEDIUM - 0.01)  # category-split cap -> level "low"
    return Confidence(
        score=round(score, 4),
        level=_level(score, HIGH, MEDIUM),
        method=CONFIDENCE_METHOD,
        calibrated=False,
        factors=(
            {"category_agreement_top5": round(agreement, 4),
             "top_job_skill_coverage": round(skill_cov, 4),
             "profile_skill_evidence": round(profile_evidence, 4),
             "score_margin_to_second": round(margin, 4)}
            if include_factors else None
        ),
    )
