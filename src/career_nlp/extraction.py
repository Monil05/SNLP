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

from .schemas import AcademicRecord, ExtractedContact, ExtractedDegree, ExtractedSkill
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
# Academic grades / subject-specific performance
# ---------------------------------------------------------------------------
# Grade extraction is intentionally evidence-based. It recognises explicit
# grade/score statements and labelled overall GPAs, but never guesses a grade
# from unrelated CV wording.
_GRADE_VALUE = (
    r"(?:\d{1,3}(?:\.\d+)?\s*(?:%|percent(?:age)?)|"
    r"\d{1,3}(?:\.\d+)?\s*(?:/|out\s+of)\s*\d{1,3}(?:\.\d+)?|"
    r"(?:grade\s*)?(?:A[+-]?|B[+-]?|C[+-]?|D[+-]?|E|F|O|EX|VG|FAIL(?:ED)?|POOR|WEAK))"
)
_POOR_WORDS = {
    "d", "d-", "d+", "e", "f", "fail", "failed", "poor", "weak", "low",
    "bad", "below average", "unsatisfactory",
}
_SUBJECT_STOPWORDS = {
    "academic", "grade", "grades", "mark", "marks", "score", "scores",
    "percentage", "result", "results", "performance", "in", "for", "of",
    "the", "and", "a", "an", "my", "got", "scored", "obtained",
}

_GPA_LABEL = r"(?:cumulative\s+grade\s+point\s+average|grade\s+point\s+average|cumulative\s+GPA|CGPA|GPA)"
_GPA_NUMBER = r"(?P<value>\d{1,2}(?:\.\d+)?)"
_GPA_SCALE_VALUE = r"4(?:\.0+)?|10(?:\.0+)?"
_GPA_SCALE = (
    rf"(?:\s*(?:/|out\s+of)\s*(?P<scale>{_GPA_SCALE_VALUE})"
    rf"|\s+on\s+(?:a\s+)?(?:(?P<scale_on>{_GPA_SCALE_VALUE})(?:[- ]point)?\s+scale"
    rf"|scale\s+of\s+(?P<scale_of>{_GPA_SCALE_VALUE})))?"
)
_GPA_SUBJECT = r"(?P<subject>[A-Za-z][A-Za-z0-9&/' -]{1,35}?)"

# Supported shapes include "GPA: 3.2/4", "GPA 3.2", "GPA 3.2 on a 4-point scale",
# "GPA in Physics: 2.1", "Physics GPA: 2.1/4", "Physics: GPA 2.1/4", and
# "3.2/4 GPA in Physics". A GPA without a declared denominator defaults to a
# 4-point scale, as explicitly requested.
_GPA_PATTERNS = (
    # Subject-first forms need to run before generic GPA patterns, otherwise
    # "Physics: GPA 1.5/4" can be mistaken for an overall GPA.
    re.compile(
        rf"^\s*{_GPA_SUBJECT}\s*[:=]\s*{_GPA_LABEL}\s*"
        rf"(?:(?:is|was|of)\s+)?{_GPA_NUMBER}{_GPA_SCALE}", re.I,
    ),
    re.compile(
        rf"^\s*{_GPA_SUBJECT}\s+(?P<label>GPA|CGPA)\s*"
        rf"(?:(?:is|was|of)\s+|[:=]\s*|\s*){_GPA_NUMBER}{_GPA_SCALE}", re.I,
    ),
    re.compile(
        rf"\b{_GPA_LABEL}\s+(?:in|for)\s+{_GPA_SUBJECT}\s*(?::|=)?\s*"
        rf"(?:(?:is|was|of)\s+)?{_GPA_NUMBER}{_GPA_SCALE}", re.I,
    ),
    re.compile(
        rf"\b{_GPA_NUMBER}{_GPA_SCALE}\s+(?P<label>GPA|CGPA)\b"
        rf"(?:\s+(?:in|for)\s+{_GPA_SUBJECT}(?=\s*(?:[,.;!?]|$)))?", re.I,
    ),
    re.compile(
        rf"\b(?:overall\s+)?{_GPA_LABEL}\b\s*"
        rf"(?:(?:is|was|of)\s+|[:=]\s*|\s*){_GPA_NUMBER}{_GPA_SCALE}"
        rf"(?:\s+(?:in|for)\s+{_GPA_SUBJECT})?", re.I,
    ),
)
_OVERALL_GPA_PREFIXES = {
    "my", "current", "overall", "cumulative", "final", "semester", "college",
    "university", "student", "candidate", "the", "academic",
}


def _clean_subject(value: str) -> str:
    """Trim a captured subject label without guessing from surrounding text."""
    subject = re.sub(r"\s+", " ", value.strip(" \t.,:;|/-"))
    subject = re.sub(r"^(?:grades?|marks?|scores?|academic results?)\s*", "", subject, flags=re.I)
    subject = re.sub(r"\s+(?:and|or|but|with|which|who|where)$", "", subject, flags=re.I)
    words = subject.split()
    while words and words[0].lower() in _SUBJECT_STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in _SUBJECT_STOPWORDS:
        words.pop()
    subject = " ".join(words).strip()
    if not subject or len(subject) > 40 or not re.search(r"[A-Za-z]", subject):
        return ""
    if subject.lower() in {"i", "my", "grade", "grades", "marks", "score", "result", "results"}:
        return ""
    # Preserve short all-caps labels (for example NLP, RAG, SQL or AI) rather
    # than converting them to visually incorrect forms such as "Nlp".
    if subject.isupper() and len(subject) <= 4 and re.fullmatch(r"[A-Z0-9+#.-]+", subject):
        return subject
    return subject.title()


def _grade_details(
    value: str,
    explicit_poor: bool = False,
    poor_threshold_percent: float = 50.0,
) -> Tuple[Optional[str], Optional[float], bool, Optional[float]]:
    """Return display grade, normalized score, poor flag, and original scale."""
    raw = re.sub(r"^grade\s+", "", value.strip(), flags=re.I).strip(" .,:;()")
    raw = re.sub(r"\s+", "", raw)
    score_percent: Optional[float] = None
    scale_max: Optional[float] = None
    fraction = re.fullmatch(r"(\d+(?:\.\d+)?)(?:/|outof)(\d+(?:\.\d+)?)", raw, re.I)
    percentage = re.fullmatch(r"(\d+(?:\.\d+)?)(?:%|percent(?:age)?)", raw, re.I)
    if percentage:
        numeric_percent = float(percentage.group(1))
        display_grade = f"{numeric_percent:g}%"
        # Keep impossible percentages visible, but do not clamp 135% to 100%
        # and then treat that invented value as valid evidence.
        if numeric_percent > 100.0:
            return display_grade, None, explicit_poor, None
        score_percent = numeric_percent
    elif fraction:
        score_value, scale_value = float(fraction.group(1)), float(fraction.group(2))
        display_grade = f"{score_value:g}/{scale_value:g}"
        if scale_value <= 0 or score_value > scale_value:
            return display_grade, None, explicit_poor, scale_value if scale_value > 0 else None
        scale_max = scale_value
        score_percent = score_value / scale_max * 100.0
    else:
        display_grade = raw.upper()
    if score_percent is not None:
        score_percent = round(score_percent, 2)
        return display_grade, score_percent, explicit_poor or score_percent <= poor_threshold_percent, scale_max
    grade = display_grade.upper()
    is_poor = explicit_poor or grade.lower() in _POOR_WORDS
    return grade or ("poor" if explicit_poor else None), None, is_poor, scale_max


def extract_academic_records(
    text: str,
    poor_threshold_percent: float = 50.0,
) -> List[AcademicRecord]:
    """Extract explicitly stated subject grades, percentages, and GPA scores.

    Supported examples include ``Physics: 35%``, ``Physics 42 percent``,
    ``Physics score 45 out of 100``, ``Maths: 4/10``, ``Physics grade: D``,
    ``I scored 38% in Physics``, ``Physics: GPA 1.5/4``, ``CGPA: 7.8/10``,
    and ``GPA: 2.1``. An unscaled GPA defaults to a 4-point scale when its
    value is <= 4, and to a 10-point scale when its value is > 4. Numeric
    values are normalised to 0-100 for comparison; values at or below
    ``poor_threshold_percent`` and D/E/F/fail labels are marked poor. Overall
    GPA is reported separately and is never treated as a subject-specific
    weakness for JD matching.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    if not 0.0 <= poor_threshold_percent <= 100.0:
        raise ValueError("poor_threshold_percent must be between 0 and 100")

    found: Dict[str, AcademicRecord] = {}

    def add(
        subject_value: str,
        grade_value: str,
        evidence: str,
        explicit_poor: bool = False,
        *,
        scope: str = "subject",
        score_percent_override: Optional[float] = None,
        scale_max_override: Optional[float] = None,
        grade_override: Optional[str] = None,
        confidence_override: Optional[float] = None,
    ) -> None:
        subject = _clean_subject(subject_value) if scope == "subject" else "Overall GPA"
        if not subject:
            return

        if score_percent_override is None and grade_override is not None:
            # Keep an explicitly stated but out-of-range GPA visible without
            # normalising it or treating it as evidence of poor performance.
            grade = grade_override
            score_percent = None
            scale_max = scale_max_override
            is_poor = explicit_poor
        elif score_percent_override is None:
            grade, score_percent, is_poor, inferred_scale = _grade_details(
                grade_value, explicit_poor, poor_threshold_percent
            )
            scale_max = scale_max_override if scale_max_override is not None else inferred_scale
        else:
            grade = grade_override or grade_value
            score_percent = round(max(0.0, min(100.0, score_percent_override)), 2)
            scale_max = scale_max_override
            is_poor = explicit_poor or score_percent <= poor_threshold_percent

        key_subject = re.sub(r"[^a-z0-9]+", " ", subject.lower()).strip()
        key = f"{scope}:{key_subject}"
        record = AcademicRecord(
            subject=subject,
            scope=scope,
            grade=grade,
            score_percent=score_percent,
            scale_max=scale_max,
            is_poor=is_poor,
            evidence=re.sub(r"\s+", " ", evidence).strip()[:240],
            confidence=confidence_override or (0.96 if scope == "overall" else (
                0.94 if explicit_poor or score_percent is not None else 0.88
            )),
        )
        current = found.get(key)
        # When a subject appears multiple times, keep the more cautionary result
        # rather than allowing a later good score to erase an explicit failure.
        if current is None or (record.is_poor and not current.is_poor):
            found[key] = record

    def parse_gpa(segment: str) -> Optional[Tuple[int, int]]:
        """Add one GPA record and return its matched span, if a GPA is present."""
        for pattern in _GPA_PATTERNS:
            match = pattern.search(segment)
            if not match:
                continue
            value = float(match.group("value"))
            groups = match.groupdict()
            explicit_scale = next(
                (groups.get(name) for name in ("scale", "scale_on", "scale_of") if groups.get(name)),
                None,
            )
            if explicit_scale:
                scale = float(explicit_scale)
            else:
                # Use the usual 4-point scale for values through 4. Values
                # above 4 are interpreted as a 10-point GPA, unless a scale
                # was explicitly stated in the text.
                scale = 4.0 if value <= 4.0 else 10.0

            subject_value = match.groupdict().get("subject")
            subject_text = (subject_value or "").strip()
            # "My GPA", "Overall GPA", and similar leads describe a total GPA,
            # not a GPA in a particular subject.
            if subject_text.lower() in _OVERALL_GPA_PREFIXES:
                subject_text = ""
            scope = "subject" if subject_text else "overall"
            grade_display = f"GPA {value:g}/{scale:g}"
            evidence = match.group(0).strip()
            if not explicit_scale:
                if scale == 4.0:
                    evidence += " (scale not stated; assumed a 4-point GPA scale)"
                else:
                    evidence += (
                        " (scale not stated; assumed a 10-point GPA scale "
                        "because the value exceeds 4)"
                    )

            # Do not silently drop an explicitly reported GPA that exceeds its
            # stated/assumed scale. Preserve it for review, but do not infer a
            # percentage or classify it as poor from an invalid value.
            if value > scale or scale <= 0:
                evidence += " (value outside the scale; not normalised)"
                add(
                    subject_text,
                    grade_display,
                    evidence,
                    scope=scope,
                    scale_max_override=scale,
                    grade_override=grade_display,
                    confidence_override=0.75,
                )
                return match.span()

            score_percent = value / scale * 100.0
            add(
                subject_text,
                grade_display,
                evidence,
                scope=scope,
                score_percent_override=score_percent,
                scale_max_override=scale,
                grade_override=grade_display,
                confidence_override=0.96,
            )
            return match.span()
        return None

    # Keep line/section boundaries because CVs often list one subject per line.
    chunks: List[str] = []
    for line in re.split(r"[\r\n]+", text):
        for part in re.split(r"[;|]+", line):
            chunks.extend(
                sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", part) if sentence.strip()
            )
    if not chunks:
        chunks = [text]

    subject_first = re.compile(
        rf"^\s*(?P<subject>[A-Za-z][A-Za-z0-9&/' -]{{1,35}}?)\s*"
        rf"(?:(?:academic\s+)?(?:grade|marks?|score|percentage|result)\s*)?"
        rf"(?::|=|\s-\s)\s*(?P<value>{_GRADE_VALUE})\s*[.,]?\s*$", re.I,
    )
    subject_with_label = re.compile(
        rf"^\s*(?P<subject>[A-Za-z][A-Za-z&/' -]{{1,30}}?)\s+"
        rf"(?:grade|marks?|score|percentage)\s*(?:is|was|of|:|=)?\s*"
        rf"(?P<value>{_GRADE_VALUE})\s*$", re.I,
    )
    subject_fraction = re.compile(
        r"^\s*(?P<subject>[A-Za-z][A-Za-z&/' -]{1,30}?)\s+"
        r"(?P<value>\d{1,3}(?:\.\d+)?\s*/\s*\d{1,3}(?:\.\d+)?)\s*[.,]?\s*$",
        re.I,
    )
    subject_score = re.compile(
        r"^\s*(?P<subject>[A-Za-z][A-Za-z&/' -]{1,30}?)\s+"
        r"(?P<value>\d{1,3}(?:\.\d+)?\s*(?:%|percent(?:age)?))\s*[.,]?\s*$", re.I,
    )
    grade_first = re.compile(
        rf"\b(?:grade\s*[:=]\s*)?(?P<value>(?:grade\s*)?(?:A[+-]?|B[+-]?|C[+-]?|D[+-]?|E|F|O|EX|VG|FAIL(?:ED)?))"
        rf"\s+(?:grade\s+)?(?:in|for)\s+(?P<subject>[A-Za-z][A-Za-z&/' -]{{1,30}}?)"
        rf"(?=\s*(?:[,.;]|$))", re.I,
    )
    grade_in_subject = re.compile(
        rf"\b(?:grade|marks?|score|result)\s+(?:in|for)\s+"
        rf"(?P<subject>[A-Za-z][A-Za-z0-9&/' -]{{1,35}}?)"
        rf"(?:\s*(?::|=)\s*|\s+(?:is|was)\s+|\s+)"
        rf"(?P<value>{_GRADE_VALUE})(?=\s|[,.;]|$)", re.I,
    )
    score_in_subject = re.compile(
        r"\b(?P<value>\d{1,3}(?:\.\d+)?\s*(?:%|percent(?:age)?)|"
        r"\d{1,3}(?:\.\d+)?\s*(?:/|out\s+of)\s*\d{1,3}(?:\.\d+)?)"
        r"\s*(?:score|marks?|grade)?\s+(?:in|for)\s+"
        r"(?P<subject>[A-Za-z][A-Za-z&/' -]{1,30}?)(?=\s*(?:and\s+\d|[,.;]|$))",
        re.I,
    )
    explicit_poor_patterns = [
        re.compile(
            r"\b(?:poor|weak|low|bad|below[- ]average|unsatisfactory|failing|failed)\s+"
            r"(?:academic\s+)?(?:grades?|marks?|scores?|performance|results?)\s+"
            r"(?:in|for|at|on)\s+(?P<subject>[A-Za-z][A-Za-z&/' -]{1,40})", re.I,
        ),
        re.compile(
            r"\b(?P<subject>[A-Za-z][A-Za-z&/' -]{1,35}?)\s+"
            r"(?:has|had|received|got|was|is)?\s*(?:a\s+)?"
            r"(?:poor|weak|low|bad|below[- ]average|unsatisfactory|failing)\s+"
            r"(?:academic\s+)?(?:grades?|marks?|scores?|performance)\b", re.I,
        ),
    ]

    for chunk in chunks:
        segment = chunk.strip()
        # CVs often introduce inline grades with a section label, e.g.
        # "Academic record: Physics: 42%; Mathematics: 76%". Remove only
        # the leading heading so the subject/grade pair remains intact.
        segment = re.sub(
            r"^(?:(?:academic\s+)?(?:grades?|marks?|scores?|results?|records?|history|transcript))\s*[:\-]\s*",
            "",
            segment,
            flags=re.I,
        )

        # Record explicit poor-grade statements before general grade patterns.
        # Mask those spans so the final letter of a subject (e.g. the "c" in
        # "Physics") cannot be mistaken for a standalone letter grade.
        characters = list(segment)
        for pattern in explicit_poor_patterns:
            for match in pattern.finditer(segment):
                subject_value = match.group("subject")
                subjects = re.split(r"\s+(?:and|&)\s+", subject_value, flags=re.I)
                for subject in subjects:
                    add(subject, "poor", match.group(0), explicit_poor=True)
                for position in range(*match.span()):
                    characters[position] = " "
        segment = "".join(characters)
        grade_segments = [part.strip() for part in segment.split(",") if part.strip()]

        for grade_segment in grade_segments:
            # Parse GPA separately so 3.2/4 is interpreted as a GPA scale, not
            # accidentally reported as an ordinary score with a lost label.
            gpa_span = parse_gpa(grade_segment)
            if gpa_span:
                grade_segment = (grade_segment[:gpa_span[0]] + " " + grade_segment[gpa_span[1]:]).strip()
                grade_segment = re.sub(r"^(?:and|also|with)\b\s*", "", grade_segment, flags=re.I)
                if not grade_segment:
                    continue

            match = (
                subject_first.match(grade_segment)
                or subject_with_label.match(grade_segment)
                or subject_fraction.match(grade_segment)
                or subject_score.match(grade_segment)
            )
            if match:
                add(match.group("subject"), match.group("value"), grade_segment)
                continue
            match = grade_first.search(grade_segment) or grade_in_subject.search(grade_segment)
            if match:
                add(match.group("subject"), match.group("value"), grade_segment)
            for match in score_in_subject.finditer(grade_segment):
                add(match.group("subject"), match.group("value"), grade_segment)
        for pattern in explicit_poor_patterns:
            match = pattern.search(segment)
            if match:
                subjects = re.split(r"\s+(?:and|&)\s+", match.group("subject"), flags=re.I)
                for subject in subjects:
                    add(subject, "poor", segment, explicit_poor=True)

    return list(found.values())


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
