import ast
import csv
import re
import zipfile
from xml.etree import ElementTree

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


def clean_description(value):
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_title(value):
    text = clean_description(value).lower()
    text = re.sub(r"\bsr\.", "senior", text)
    text = re.sub(r"\bsr\b", "senior", text)
    text = re.sub(r"\bjr\.", "junior", text)
    text = re.sub(r"\bjr\b", "junior", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_skill(value):
    text = clean_description(value).lower()
    text = re.sub(r"\bpower\s*bi\b", "power bi", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_skills(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_skills = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw_skills = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            raw_skills = text.split(";")
        if not isinstance(raw_skills, (list, tuple, set)):
            raw_skills = [raw_skills]

    skills = []
    for skill in raw_skills:
        normalized = normalize_skill(skill)
        if normalized and normalized not in skills:
            skills.append(normalized)
    return skills


def format_skills(skills):
    return "; ".join(skills)


def parse_saved_skills(value):
    return parse_skills(value)


def build_job_text(row):
    skills = parse_saved_skills(row.get("job_skill_set", ""))
    parts = [
        clean_description(row.get("category", "")).lower(),
        clean_description(row.get("job_title", "")).lower(),
        clean_description(row.get("job_description", "")).lower(),
        " ".join(skills),
    ]
    return " ".join(part for part in parts if part).strip()


def split_jobs(jobs):
    train_jobs, test_jobs = train_test_split(
        jobs,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=jobs["category"],
    )
    return train_jobs.reset_index(drop=True), test_jobs.reset_index(drop=True)


def fit_tfidf(train_jobs):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    vectorizer.fit(train_jobs.apply(build_job_text, axis=1))
    return vectorizer


def build_skill_vocabulary(jobs):
    vocabulary = set()
    for skills in jobs["job_skill_set"]:
        vocabulary.update(parse_saved_skills(skills))
    return sorted(vocabulary, key=len, reverse=True)


def extract_skills(text, skill_vocabulary):
    normalized_text = normalize_skill(text)
    found = []
    for skill in skill_vocabulary:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, normalized_text):
            found.append(skill)
    return found


def compare_skills(student_text, job_skills, skill_vocabulary):
    student_skills = set(extract_skills(student_text, skill_vocabulary))
    required_skills = set(normalize_skill(skill) for skill in job_skills)
    matched = sorted(student_skills.intersection(required_skills))
    missing = sorted(required_skills.difference(student_skills))
    return matched, missing


def match_job_description(student_text, job_description, vectorizer, skill_vocabulary):
    job_skills = extract_skills(job_description, skill_vocabulary)
    student_text = clean_description(student_text)
    job_text = clean_description(job_description)
    cosine_score = float(cosine_similarity(
        vectorizer.transform([student_text]),
        vectorizer.transform([job_text]),
    )[0][0])
    matched, missing = compare_skills(student_text, job_skills, skill_vocabulary)
    required_skill_count = len(set(job_skills))
    skill_coverage = (
        len(matched) / required_skill_count if required_skill_count else 0.0
    )
    # Week 2 is deliberately a simple lexical baseline. Skill overlap stays
    # in the response as an explanation, but does not affect its ranking.
    final_score = cosine_score
    return {
        "match_score": round(final_score * 100, 2),
        "matched_skills": format_skills(matched),
        "missing_weak_skills": format_skills(missing),
        "recommended_learning": format_skills(missing),
    }


def score_student_against_job(student_text, job_row, vectorizer, skill_vocabulary):
    """Score one student profile against one prepared job row."""
    student_text = clean_description(student_text)
    job_text = build_job_text(job_row)
    cosine_score = float(cosine_similarity(
        vectorizer.transform([student_text]),
        vectorizer.transform([job_text]),
    )[0][0])

    required_skills = parse_saved_skills(job_row.get("job_skill_set", ""))
    student_skills = set(extract_skills(student_text, skill_vocabulary))
    required_skill_set = set(required_skills)
    matched = sorted(student_skills.intersection(required_skill_set))
    missing = sorted(required_skill_set.difference(student_skills))
    skill_coverage = (
        len(matched) / len(required_skill_set) if required_skill_set else 0.0
    )
    # Keep the baseline score purely lexical; skill coverage is explanatory.
    final_score = cosine_score
    return {
        "match_score": round(final_score * 100, 2),
        "cosine_score": round(cosine_score * 100, 2),
        "skill_coverage": round(skill_coverage * 100, 2),
        "matched_skills": format_skills(matched),
        "missing_weak_skills": format_skills(missing),
        "recommended_learning": format_skills(missing),
    }

def save_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "student_identifier",
        "degree",
        "profession",
        "student_skills",
        "student_profile_text",
        "job_description",
        "match_score",
        "matched_skills",
        "missing_weak_skills",
        "recommended_learning",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def extract_docx_text(path):
    try:
        from docx import Document

        document = Document(path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except ImportError:
        with zipfile.ZipFile(path) as archive:
            xml_content = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml_content)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        return " ".join(node.text or "" for node in root.findall(".//w:t", namespace))


def extract_pdf_text(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        print("Install pypdf to read PDF resumes.")
        return ""

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_cv_text(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix == ".docx":
        return extract_docx_text(path)
    return ""
