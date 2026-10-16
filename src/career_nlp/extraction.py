"""Rule-based information extraction (skills, degrees, contacts, experience).

Differences from the Week 3 extractor (all deliberate fixes, see
WEEK6_ASSET_ENGINEERING.md > "Changes vs Week 3"):

* Degree patterns were case-insensitive, so ordinary words were read as degrees
  ("to be" -> B.E., "contact me" -> M.E., "MS Office" -> M.S.). Ambiguous
  abbreviations (BE, ME, MS, BA, MA) now need dots (B.E.) or an explicit
  "in <subject>" / "(<subject>)" continuation, and are matched case-sensitively.
* The phone pattern matched runs of years ("2019 2020 2021"); year-only runs and
  implausible digit counts are now rejected.
* Every entity carries a rule-based confidence.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .schemas import ExtractedContact, ExtractedDegree, ExtractedSkill
from .text import SkillExtractor

# ---------------------------------------------------------------------------
# Degrees
# ---------------------------------------------------------------------------
# Unambiguous abbreviations: matched case-insensitively, dots optional.
_SAFE_DEGREES = {
    "btech": r"B\.?\s?Tech\.?",
    "mtech": r"M\.?\s?Tech\.?",
    "bsc": r"B\.?\s?Sc\.?",
    "msc": r"M\.?\s?Sc\.?",
    "bcom": r"B\.?\s?Com\.?",
    "mcom": r"M\.?\s?Com\.?",
    "bba": r"B\.?\s?B\.?\s?A\.?",
    "mba": r"M\.?\s?B\.?\s?A\.?",
    "cfa": r"CFA",
    "cpa": r"CPA",
    "phd": r"Ph\.?\s?D\.?",
}
# Ambiguous with English words (be/me/ms/ba/ma): dotted form, or uppercase
# followed by "in"/"(" so that "MS Office" and "TO BE" are not degrees.
_AMBIGUOUS_DEGREES = {
    "be": ("B.E.", r"B\.E\.", r"BE"),
    "me": ("M.E.", r"M\.E\.", r"ME"),
    "ms": ("M.S.", r"M\.S\.", r"MS"),
    "ba": ("B.A.", r"B\.A\.", r"BA"),
    "ma": ("M.A.", r"M\.A\.", r"MA"),
}
DEGREE_DISPLAY = {
    "btech": "B.Tech", "mtech": "M.Tech", "bsc": "B.Sc.", "msc": "M.Sc.",
    "bcom": "B.Com.", "mcom": "M.Com.", "bba": "BBA", "mba": "MBA",
    "cfa": "CFA", "cpa": "CPA", "phd": "Ph.D.",
}

_SPEC_STOPWORDS = {
    "with", "and", "having", "experience", "years", "year", "student",
    "graduate", "background", "from", "at", "the", "a", "an", "who",
    "currently", "seeking", "looking", "skilled", "proficient", "expert",
    "in", "of", "for", "to", "is", "was", "office", "mba", "btech", "mtech",
}
_SPEC_RE = re.compile(r"^\s*[,:\-]?\s*(?:in|of)?\s*((?:[A-Za-z][A-Za-z&]*\s*){1,4})")
_SPEC_PAREN_RE = re.compile(r"^\s*\(\s*([A-Za-z][A-Za-z& ]{1,40}?)\s*\)")


def _specialization(remainder: str) -> str:
    paren = _SPEC_PAREN_RE.match(remainder)
    if paren:
        return paren.group(1).strip()
    m = _SPEC_RE.match(remainder)
    if not m:
        return ""
    kept: List[str] = []
    for word in m.group(1).split():
        if word.lower() in _SPEC_STOPWORDS:
            break
        kept.append(word)
    return " ".join(kept)


def extract_degrees(text: str) -> List[ExtractedDegree]:
    found: Dict[str, ExtractedDegree] = {}

    def add(label: str, remainder: str, base_conf: float) -> None:
        spec = _specialization(remainder)
        if spec.islower():
            spec = spec.title()  # 'computer science' -> 'Computer Science'
        normalized = f"{label} in {spec}" if spec else label
        key = normalized.lower()
        if key in found:
            return
        conf = min(1.0, base_conf + (0.05 if spec else 0.0))
        found[key] = ExtractedDegree(
            text=normalized, degree=label, specialization=spec or None,
            confidence=round(conf, 2),
        )

    for key, pat in _SAFE_DEGREES.items():
        for m in re.finditer(r"(?<![A-Za-z])" + pat + r"(?![A-Za-z])", text, flags=re.I):
            # professional certifications (CFA/CPA) have no "specialization"
            add(DEGREE_DISPLAY[key], "" if key in {"cfa", "cpa"} else text[m.end():], 0.90)

    for key, (label, dotted, bare) in _AMBIGUOUS_DEGREES.items():
        for m in re.finditer(r"(?<![A-Za-z])" + dotted + r"(?![A-Za-z])", text):
            add(label, text[m.end():], 0.85)
        # Bare uppercase form only when followed by "in <subject>" or "(<subject>)".
        for m in re.finditer(r"(?<![A-Za-z])" + bare + r"(?=\s+in\s+[A-Za-z]|\s*\()", text):
            add(label, text[m.end():], 0.75)
    return list(found.values())


# ---------------------------------------------------------------------------
# Contacts / experience
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
_YEARS_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*\+?\s*(?:years?|yrs?)\b",
    re.I,
)
_WORD_NUM = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])}


def _is_year_run(candidate: str) -> bool:
    chunks = [c for c in re.split(r"[\s.\-()]+", candidate) if c]
    return bool(chunks) and all(
        c.isdigit() and len(c) == 4 and 1900 <= int(c) <= 2100 for c in chunks
    )


def extract_emails(text: str) -> List[ExtractedContact]:
    seen = list(dict.fromkeys(EMAIL_RE.findall(text)))
    return [ExtractedContact(value=e, confidence=0.98) for e in seen]


def extract_phones(text: str) -> List[ExtractedContact]:
    out: List[ExtractedContact] = []
    for cand in dict.fromkeys(m.group(0).strip() for m in _PHONE_RE.finditer(text)):
        digits = re.sub(r"\D", "", cand)
        if not 10 <= len(digits) <= 13 or _is_year_run(cand):
            continue
        conf = 0.9 if cand.startswith("+") else (0.8 if len(digits) == 10 else 0.65)
        out.append(ExtractedContact(value=cand, confidence=conf))
    return out


def extract_experience(text: str) -> Tuple[List[str], Optional[float]]:
    mentions = list(dict.fromkeys(m.group(0) for m in _YEARS_RE.finditer(text)))
    values: List[float] = []
    for mention in mentions:
        token = re.match(r"[A-Za-z]+|\d+(?:\.\d+)?", mention)
        if not token:
            continue
        raw = token.group(0).lower()
        values.append(float(_WORD_NUM[raw]) if raw in _WORD_NUM else float(raw))
    return mentions, (max(values) if values else None)


# ---------------------------------------------------------------------------
# Skills with rule-based confidence
# ---------------------------------------------------------------------------
def skill_confidence(name: str, mentions: int) -> float:
    """Heuristic confidence that a vocabulary hit is a genuine skill mention.

    exact word-boundary vocabulary hit .......... 0.90
    repeated mention (>=2) ...................... +0.05
    multi-word skill (specific phrase) .......... +0.05
    very short token (<=2 chars, e.g. "r", "c") . -0.25  (high collision risk)
    short token (3 chars, e.g. "sap", "tax") .... -0.05
    """
    conf = 0.90
    if mentions >= 2:
        conf += 0.05
    if len(name.split()) >= 2:
        conf += 0.05
    if len(name) <= 2:
        conf -= 0.25
    elif len(name) == 3:
        conf -= 0.05
    return round(max(0.0, min(1.0, conf)), 2)


def extract_skills(text: str, extractor: SkillExtractor) -> List[ExtractedSkill]:
    return [
        ExtractedSkill(name=n, mentions=c, confidence=skill_confidence(n, c))
        for n, c in extractor.find(text).items()
    ]
