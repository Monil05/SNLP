"""Text normalisation and skill-vocabulary utilities.

Behaviour intentionally mirrors ``week2.baseline`` (clean_description,
normalize_skill, parse_skills, build_job_text) so scores stay comparable with
Week 3. The one performance change: skill matching pre-filters with a C-level
substring test before running the word-boundary regex, which keeps the exact
same results while being much faster on a ~5k-skill vocabulary.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_skill(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"\bpower\s*bi\b", "power bi", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_skills(value: Any) -> List[str]:
    """Parse a ``job_skill_set`` cell (python-list string, ';'-separated string
    or real list) into a de-duplicated list of normalised skills."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw: Iterable = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            raw = text.split(";")
        if not isinstance(raw, (list, tuple, set)):
            raw = [raw]
    skills: List[str] = []
    for item in raw:
        norm = normalize_skill(item)
        if norm and norm not in skills:
            skills.append(norm)
    return skills


def format_skills(skills: Sequence[str]) -> str:
    """Legacy Week 3 string form: 'a; b; c'."""
    return "; ".join(skills)


def build_job_text(row: Dict[str, Any]) -> str:
    skills = parse_skills(row.get("job_skill_set", ""))
    parts = [
        clean_text(row.get("category", "")).lower(),
        clean_text(row.get("job_title", "")).lower(),
        clean_text(row.get("job_description", "")).lower(),
        " ".join(skills),
    ]
    return " ".join(p for p in parts if p).strip()


_TITLE_LEVEL_RE = re.compile(r"\s+(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\s*$", re.I)


def clean_job_title(title: str) -> str:
    """Strip parenthetical focus text, requisition codes ('- r7269') and
    trailing Roman-numeral levels from dataset job titles."""
    cleaned = clean_text(title)
    cleaned = re.sub(r"\s*\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\s*-\s*r\d+\s*$", "", cleaned, flags=re.I)
    cleaned = _TITLE_LEVEL_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


class SkillExtractor:
    """Exact, word-boundary skill matcher over a fixed vocabulary."""

    def __init__(self, vocabulary: Iterable[str]):
        # Longest first, same as week2.build_skill_vocabulary.
        self.vocabulary: List[str] = sorted(set(vocabulary), key=len, reverse=True)
        self._patterns: Dict[str, "re.Pattern[str]"] = {
            s: re.compile(r"(?<!\w)" + re.escape(s) + r"(?!\w)") for s in self.vocabulary
        }

    def __len__(self) -> int:
        return len(self.vocabulary)

    def find(self, text: str) -> Dict[str, int]:
        """Return {skill: mention_count} for every vocabulary skill present."""
        norm = normalize_skill(text)
        found: Dict[str, int] = {}
        for skill in self.vocabulary:
            if skill not in norm:  # cheap necessary condition
                continue
            count = len(self._patterns[skill].findall(norm))
            if count:
                found[skill] = count
        return found

    def extract(self, text: str) -> List[str]:
        return list(self.find(text))

    @classmethod
    def from_jobs(cls, jobs: pd.DataFrame) -> "SkillExtractor":
        vocab = set()
        for cell in jobs["job_skill_set"]:
            vocab.update(parse_skills(cell))
        return cls(vocab)
