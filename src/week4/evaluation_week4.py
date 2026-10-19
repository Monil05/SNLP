"""Week 4 — Baseline vs Improved NLP model comparison.

Baseline:
    Week 2 TF-IDF cosine similarity fitted on the training-job split

Improved:
    Week 3 hybrid: skill coverage + Sentence-Transformer semantic similarity
    + category compatibility from a classifier trained on the train split

Both models are evaluated on exactly the same synthetic students and the
same held-out test job set. Relevance is defined by the student's expected
job category, matching the existing baseline evaluation methodology.

Run from the project root:
    python "base version/src/evaluation_week4.py"

The improved model requires sentence-transformers. Its model weights are
cached by week3/nlp_v1.py under:
    base version/data/sentence_transformer_model/
so the weights are downloaded only on the first run.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
TEST_DIR = BASE_DIR / "test"
TRAIN_JOBS_PATH = DATA_DIR / "train_jobs.csv"
TEST_JOBS_PATH = DATA_DIR / "test_jobs.csv"
STUDENTS_PATH = TEST_DIR / "test_students.csv"

DETAIL_PATH = TEST_DIR / "week4_comparison_results.csv"
SUMMARY_PATH = TEST_DIR / "week4_comparison_summary.csv"
MODEL_PATH = TEST_DIR / "week4_model_comparison.csv"

K = 5

EXPECTED_CATEGORIES = {
    "student_01": "INFORMATION-TECHNOLOGY",
    "student_02": "HR",
    "student_03": "FINANCE",
    "student_04": "SALES",
    "student_05": "INFORMATION-TECHNOLOGY",
    "student_06": "BUSINESS-DEVELOPMENT",
    "student_07": "INFORMATION-TECHNOLOGY",
    "student_08": "FINANCE",
    "student_09": "SALES",
    "student_10": "HR",
    "student_11": "INFORMATION-TECHNOLOGY",
    "student_12": "SALES",
    "student_13": "FINANCE",
    "student_14": "INFORMATION-TECHNOLOGY",
    "student_15": "HR",
    "student_16": "SALES",
}

# Import the unchanged Week 2 baseline implementation.
sys.path.insert(0, str(BASE_DIR / "src" / "week2"))
from baseline import (  # noqa: E402
    build_job_text,
    build_skill_vocabulary,
    extract_skills,
    fit_tfidf,
    parse_saved_skills,
)

# Import the Week 3 improved model.
sys.path.insert(0, str(BASE_DIR / "src" / "week3"))
from nlp_v1 import V1NLP  # noqa: E402


def reciprocal_rank(categories, expected_category):
    for rank, category in enumerate(categories, start=1):
        if category == expected_category:
            return 1.0 / rank
    return 0.0


def validate_inputs(train_jobs, test_jobs, students):
    required_job_cols = {
        "job_id", "category", "job_title", "job_description", "job_skill_set"
    }
    for name, df in [("training", train_jobs), ("test", test_jobs)]:
        missing = required_job_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name.title()} jobs missing columns: {sorted(missing)}")

    required_student_cols = {"student_id", "degree", "profession", "skills"}
    missing = required_student_cols - set(students.columns)
    if missing:
        raise ValueError(f"Students missing columns: {sorted(missing)}")

    unknown = set(students["student_id"]) - set(EXPECTED_CATEGORIES)
    if unknown:
        raise ValueError(f"No expected category defined for: {sorted(unknown)}")


def evaluate_baseline(test_jobs, students, vectorizer, skill_vocabulary):
    """Evaluate the existing Week 2 baseline on the held-out test jobs."""
    job_vectors = vectorizer.transform(test_jobs.apply(build_job_text, axis=1))
    details = []
    summaries = []

    for _, student in students.iterrows():
        student_id = student["student_id"]
        expected = EXPECTED_CATEGORIES[student_id]
        profile_text = " ".join(
            [str(student["degree"]), str(student["profession"]), str(student["skills"])]
        ).strip()

        student_vector = vectorizer.transform([profile_text])
        cosine_scores = cosine_similarity(student_vector, job_vectors)[0]
        student_skills = set(extract_skills(profile_text, skill_vocabulary))

        rows = []
        for i, (_, job) in enumerate(test_jobs.iterrows()):
            required = set(parse_saved_skills(job["job_skill_set"]))
            matched = student_skills.intersection(required)
            skill_coverage = len(matched) / len(required) if required else 0.0
            semantic = float(cosine_scores[i])
            score = semantic

            rows.append({
                "student_id": student_id,
                "expected_category": expected,
                "job_id": str(job["job_id"]),
                "category": str(job["category"]),
                "job_title": str(job["job_title"]),
                "match_score": round(score * 100, 2),
                "skill_coverage": round(skill_coverage * 100, 2),
                "semantic_similarity": round(semantic * 100, 2),
                "matched_skills": "; ".join(sorted(matched)),
            })

        ranked = pd.DataFrame(rows).sort_values(
            ["match_score", "semantic_similarity"],
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)

        ranked["rank"] = np.arange(1, len(ranked) + 1)
        top = ranked.head(K)
        relevant = ranked["category"].eq(expected)
        relevant_top = int(top["category"].eq(expected).sum())
        total_relevant = int(relevant.sum())

        details.extend(
            {"model": "baseline_tfidf", **row}
            for row in ranked.head(K).to_dict("records")
        )
        summaries.append({
            "model": "baseline_tfidf",
            "student_id": student_id,
            "expected_category": expected,
            "precision_at_5": relevant_top / K,
            "recall_at_5": relevant_top / total_relevant if total_relevant else 0.0,
            "mrr": reciprocal_rank(ranked["category"].tolist(), expected),
            "first_relevant_rank": (
                int(ranked.index[relevant][0] + 1) if relevant.any() else ""
            ),
            "top_1_category": ranked.iloc[0]["category"],
            "top_1_job_title": ranked.iloc[0]["job_title"],
        })

    return pd.DataFrame(details), pd.DataFrame(summaries)


def evaluate_improved(test_jobs, students):
    """Evaluate the Week 3 Sentence-Transformer improved representation.

    The production V1NLP pipeline keeps its searchable index on the full
    cleaned dataset. For a fair comparison, this evaluator ranks only the
    same held-out test_jobs used by the baseline. The pre-trained
    Sentence-Transformer itself is not fitted on the test labels.
    """
    try:
        improved = V1NLP()
    except RuntimeError as exc:
        raise RuntimeError(
            "\nThe improved evaluation requires the Sentence-Transformer "
            "dependency/model.\n"
            "Install dependencies first:\n"
            "  pip install -r \"base version/requirements.txt\"\n"
            "Then run this script again.\n\n"
            f"Original error: {exc}"
        ) from exc

    job_id_to_index = {
        job_id: i for i, job_id in enumerate(improved._job_ids)
    }
    missing_ids = [
        str(job_id) for job_id in test_jobs["job_id"]
        if str(job_id) not in job_id_to_index
    ]
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} test job IDs are missing from the improved model index."
        )

    test_indices = [job_id_to_index[str(job_id)] for job_id in test_jobs["job_id"]]
    test_job_vectors = improved._job_index_vectors[test_indices]

    raw_profiles = [
        " ".join([str(s.degree), str(s.profession), str(s.skills)]).strip()
        for s in students.itertuples()
    ]
    profiles = [improved.derive_student_profile(text) for text in raw_profiles]
    # Batch encoding avoids repeatedly paying Sentence-Transformer overhead.
    student_vectors = improved._embed_texts([p["semantic_text"] for p in profiles])
    semantic_percents = np.clip(
        cosine_similarity(student_vectors, test_job_vectors) * 100.0, 0.0, 100.0
    )
    classifier_vectors = improved.category_vectorizer.transform(raw_profiles)
    category_probabilities = improved.category_classifier.predict_proba(classifier_vectors)
    class_index = {label: i for i, label in enumerate(improved.category_classifier.classes_)}

    details = []
    summaries = []
    for student_index, (_, student) in enumerate(students.iterrows()):
        student_id = student["student_id"]
        expected = EXPECTED_CATEGORIES[student_id]
        student_skills = set(profiles[student_index]["skills"])
        rows = []
        for job_index, (_, job) in enumerate(test_jobs.iterrows()):
            required = set(parse_saved_skills(job["job_skill_set"]))
            matched = student_skills.intersection(required)
            skill_coverage = len(matched) / len(required) if required else 0.0
            semantic = float(semantic_percents[student_index, job_index])
            category = str(job["category"])
            category_compatibility = float(
                category_probabilities[student_index, class_index[category]] * 100.0
            )
            score = improved._combine_score(
                skill_coverage * 100.0, semantic, category_compatibility
            )
            rows.append({
                "student_id": student_id,
                "expected_category": expected,
                "job_id": str(job["job_id"]),
                "category": category,
                "job_title": str(job["job_title"]),
                "match_score": round(score, 2),
                "skill_coverage": round(skill_coverage * 100, 2),
                "semantic_similarity": round(semantic, 2),
                "category_compatibility": round(category_compatibility, 2),
                "matched_skills": "; ".join(sorted(matched)),
            })

        ranked = pd.DataFrame(rows).sort_values(
            ["match_score", "semantic_similarity", "skill_coverage"],
            ascending=False, kind="mergesort",
        ).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        top = ranked.head(K)
        relevant = ranked["category"].eq(expected)
        relevant_top = int(top["category"].eq(expected).sum())
        total_relevant = int(relevant.sum())
        details.extend({"model": "improved_sentence_transformer", **row}
                       for row in ranked.head(K).to_dict("records"))
        summaries.append({
            "model": "improved_sentence_transformer", "student_id": student_id,
            "expected_category": expected, "precision_at_5": relevant_top / K,
            "recall_at_5": relevant_top / total_relevant if total_relevant else 0.0,
            "mrr": reciprocal_rank(ranked["category"].tolist(), expected),
            "first_relevant_rank": int(ranked.index[relevant][0] + 1) if relevant.any() else "",
            "top_1_category": ranked.iloc[0]["category"],
            "top_1_job_title": ranked.iloc[0]["job_title"],
        })

    return pd.DataFrame(details), pd.DataFrame(summaries)


def build_overall(summary):
    rows = []
    for model, group in summary.groupby("model", sort=False):
        rows.append({
            "model": model,
            "student_id": "OVERALL",
            "expected_category": "",
            "precision_at_5": group["precision_at_5"].mean(),
            "recall_at_5": group["recall_at_5"].mean(),
            "mrr": group["mrr"].mean(),
            "first_relevant_rank": (
                group["first_relevant_rank"].replace("", np.nan).astype(float).mean()
            ),
            "top_1_category": "",
            "top_1_job_title": "",
        })
    return pd.DataFrame(rows)


def build_comparison(overall):
    baseline = overall[overall["model"] == "baseline_tfidf"].iloc[0]
    improved = overall[overall["model"] == "improved_sentence_transformer"].iloc[0]

    rows = []
    for metric in ["precision_at_5", "recall_at_5", "mrr"]:
        b = float(baseline[metric])
        i = float(improved[metric])
        rows.append({
            "metric": metric,
            "baseline_tfidf": b,
            "improved_sentence_transformer": i,
            "absolute_change": i - b,
            "percentage_change": ((i - b) / b * 100.0) if b else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    for path in [TRAIN_JOBS_PATH, TEST_JOBS_PATH, STUDENTS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}. Run prepare_data.py first."
            )

    train_jobs = pd.read_csv(TRAIN_JOBS_PATH).fillna("")
    test_jobs = pd.read_csv(TEST_JOBS_PATH).fillna("")
    students = pd.read_csv(STUDENTS_PATH).fillna("")
    validate_inputs(train_jobs, test_jobs, students)

    # Fit the baseline only on training jobs.  Even though TF-IDF is
    # unsupervised, fitting IDF weights or the vocabulary on held-out job
    # descriptions would leak test-distribution information into the
    # comparison.  The test set is used only for ranking and metrics.
    baseline_vectorizer = fit_tfidf(train_jobs)
    baseline_vocab = build_skill_vocabulary(train_jobs)
    baseline_details, baseline_summary = evaluate_baseline(
        test_jobs, students, baseline_vectorizer, baseline_vocab
    )

    # Improved: hybrid Sentence-Transformer, skill, and train-split category
    # compatibility signals.
    improved_details, improved_summary = evaluate_improved(test_jobs, students)

    details = pd.concat([baseline_details, improved_details], ignore_index=True)
    summary = pd.concat([baseline_summary, improved_summary], ignore_index=True)
    overall = build_overall(summary)
    comparison = build_comparison(overall)

    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(DETAIL_PATH, index=False)
    pd.concat([summary, overall], ignore_index=True).to_csv(SUMMARY_PATH, index=False)
    comparison.to_csv(MODEL_PATH, index=False)

    print("\nWEEK 4 — BASELINE VS IMPROVED MODEL")
    print("=" * 60)
    print("Evaluation set: 16 synthetic students × 234 held-out test jobs")
    print("Relevance label: expected job category (proxy label)")
    print()
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Detailed top-5 results: {DETAIL_PATH}")
    print(f"Per-student metrics:     {SUMMARY_PATH}")
    print(f"Model comparison:         {MODEL_PATH}")


if __name__ == "__main__":
    main()
