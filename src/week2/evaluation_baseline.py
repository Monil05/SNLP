from pathlib import Path

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from baseline import (
    build_job_text,
    build_skill_vocabulary,
    extract_skills,
    fit_tfidf,
    parse_saved_skills,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
TEST_DIR = BASE_DIR / "test"
TRAIN_JOBS_PATH = DATA_DIR / "train_jobs.csv"
TEST_JOBS_PATH = DATA_DIR / "test_jobs.csv"
STUDENTS_PATH = TEST_DIR / "test_students.csv"
OUTPUT_PATH = TEST_DIR / "initial_evaluation_results.csv"
SUMMARY_PATH = TEST_DIR / "initial_evaluation_summary.csv"

# Ground-truth proxy labels for the synthetic evaluation profiles.
# These labels define which job category is considered relevant for each student.
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

K = 5


def validate_inputs(train_jobs, test_jobs, students):
    required_job_cols = {"job_id", "category", "job_title", "job_description", "job_skill_set"}
    missing_train = required_job_cols - set(train_jobs.columns)
    missing_test = required_job_cols - set(test_jobs.columns)
    required_student_cols = {"student_id", "degree", "profession", "skills"}
    missing_students = required_student_cols - set(students.columns)

    if missing_train:
        raise ValueError(f"Training jobs missing columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Test jobs missing columns: {sorted(missing_test)}")
    if missing_students:
        raise ValueError(f"Students missing columns: {sorted(missing_students)}")

    unknown_students = set(students["student_id"]) - set(EXPECTED_CATEGORIES)
    if unknown_students:
        raise ValueError(f"No expected category defined for: {sorted(unknown_students)}")


def reciprocal_rank(categories, expected_category):
    for rank, category in enumerate(categories, start=1):
        if category == expected_category:
            return 1.0 / rank
    return 0.0


def evaluate_student(student, test_jobs, vectorizer, skill_vocabulary, job_vectors):
    profile_text = " ".join(
        [str(student["degree"]), str(student["profession"]), str(student["skills"])]
    ).strip()
    expected_category = EXPECTED_CATEGORIES[student["student_id"]]

    student_vector = vectorizer.transform([profile_text])
    cosine_scores = cosine_similarity(student_vector, job_vectors)[0]
    student_skills = set(extract_skills(profile_text, skill_vocabulary))

    rows = []
    for i, (_, job) in enumerate(test_jobs.iterrows()):
        required_skills = set(parse_saved_skills(job["job_skill_set"]))
        matched = sorted(student_skills.intersection(required_skills))
        missing = sorted(required_skills.difference(student_skills))
        skill_coverage = len(matched) / len(required_skills) if required_skills else 0.0
        # Week 2 baseline ranking is TF-IDF cosine similarity only.
        final_score = float(cosine_scores[i])
        rows.append({
            "student_id": student["student_id"],
            "expected_category": expected_category,
            "job_id": job["job_id"],
            "category": job["category"],
            "job_title": job["job_title"],
            "match_score": round(final_score * 100, 2),
            "cosine_score": round(float(cosine_scores[i]) * 100, 2),
            "skill_coverage": round(skill_coverage * 100, 2),
            "matched_skills": "; ".join(matched),
            "missing_weak_skills": "; ".join(missing),
            "recommended_learning": "; ".join(missing),
        })

    return pd.DataFrame(rows).sort_values(
        ["match_score", "cosine_score"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


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

    # Fit TF-IDF on training jobs only.  Although it is unsupervised, fitting
    # vocabulary or IDF statistics on held-out job descriptions would leak
    # test-distribution information into this Week 2 evaluation.
    vectorizer = fit_tfidf(train_jobs)
    skill_vocabulary = build_skill_vocabulary(train_jobs)
    job_texts = test_jobs.apply(build_job_text, axis=1)
    job_vectors = vectorizer.transform(job_texts)

    all_ranked = []
    summary = []

    for _, student in students.iterrows():
        ranked = evaluate_student(student, test_jobs, vectorizer, skill_vocabulary, job_vectors)
        top_k = ranked.head(K)
        relevant_top_k = (top_k["category"] == top_k["expected_category"]).sum()
        total_relevant = (ranked["category"] == ranked["expected_category"]).sum()

        precision_at_k = relevant_top_k / len(top_k) if len(top_k) else 0.0
        recall_at_k = relevant_top_k / total_relevant if total_relevant else 0.0
        mrr = reciprocal_rank(ranked["category"].tolist(), ranked["expected_category"].iloc[0])

        all_ranked.append(ranked)
        summary.append(
            {
                "student_id": student["student_id"],
                "expected_category": ranked["expected_category"].iloc[0],
                "test_jobs": len(ranked),
                "relevant_jobs": int(total_relevant),
                "precision_at_5": round(precision_at_k, 4),
                "recall_at_5": round(recall_at_k, 4),
                "mrr": round(mrr, 4),
            }
        )

    results = pd.concat(all_ranked, ignore_index=True)
    summary_df = pd.DataFrame(summary)

    overall = pd.DataFrame(
        [{
            "student_id": "OVERALL",
            "expected_category": "",
            "test_jobs": len(test_jobs),
            "relevant_jobs": "",
            "precision_at_5": round(summary_df["precision_at_5"].mean(), 4),
            "recall_at_5": round(summary_df["recall_at_5"].mean(), 4),
            "mrr": round(summary_df["mrr"].mean(), 4),
        }]
    )
    summary_df = pd.concat([summary_df, overall], ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("Initial baseline evaluation complete")
    print(f"Ranked results: {OUTPUT_PATH}")
    print(f"Summary metrics: {SUMMARY_PATH}")
    print("\nOverall metrics")
    print(f"Precision@5: {overall.iloc[0]['precision_at_5']:.4f}")
    print(f"Recall@5:    {overall.iloc[0]['recall_at_5']:.4f}")
    print(f"MRR:         {overall.iloc[0]['mrr']:.4f}")


if __name__ == "__main__":
    main()
