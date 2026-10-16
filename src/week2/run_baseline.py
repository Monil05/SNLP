from pathlib import Path

import pandas as pd

from baseline import (
    build_skill_vocabulary,
    extract_cv_text,
    fit_tfidf,
    match_job_description,
    save_results,
    split_jobs,
)
from prepare_data import main as prepare_data


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CV_DIR = BASE_DIR / "cv"
RESULTS_DIR = BASE_DIR / "test"
CLEANED_JOBS_PATH = DATA_DIR / "cleaned_jobs.csv"
TRAIN_JOBS_PATH = DATA_DIR / "train_jobs.csv"


def load_jobs():
    if not CLEANED_JOBS_PATH.exists():
        prepare_data()
    return pd.read_csv(CLEANED_JOBS_PATH)


def get_manual_student():
    degree = input("Degree: ").strip()
    profession = input("Profession: ").strip()
    skills = input("Skills: ").strip()
    other_text = input("Other profile text (optional): ").strip()
    profile_text = " ".join([degree, profession, skills, other_text]).strip()
    return [
        {
            "student_identifier": "manual_student",
            "degree": degree,
            "profession": profession,
            "student_skills": skills,
            "student_profile_text": profile_text,
        }
    ]


def get_cv_students():
    CV_DIR.mkdir(parents=True, exist_ok=True)
    cv_files = sorted(
        file for file in CV_DIR.iterdir() if file.suffix.lower() in {".pdf", ".docx"}
    )
    if not cv_files:
        print(f"No CV files found in {CV_DIR}")
        return []

    students = []
    for cv_file in cv_files:
        text = extract_cv_text(cv_file).strip()
        if not text:
            print(f"Could not extract text from {cv_file.name}")
            continue
        students.append(
            {
                "student_identifier": cv_file.stem,
                "degree": "",
                "profession": "",
                "student_skills": "",
                "student_profile_text": text,
            }
        )
    return students


def choose_students():
    print("Choose student input:")
    print("1. Type student profile")
    print("2. Use CV from cv/ folder")
    print("3. Run test_students.csv")
    choice = input("Enter choice: ").strip()
    if choice == "1":
        return get_manual_student()
    if choice == "2":
        return get_cv_students()
    if choice == "3":
        return get_test_students()
    print("Invalid choice.")
    return []


def get_test_students():
    students_path = BASE_DIR / "test" / "test_students.csv"
    if not students_path.exists():
        prepare_data()
    students = []
    for _, row in pd.read_csv(students_path).fillna("").iterrows():
        profile_text = " ".join(
            [str(row["degree"]), str(row["profession"]), str(row["skills"])]
        ).strip()
        students.append(
            {
                "student_identifier": row["student_id"],
                "degree": row["degree"],
                "profession": row["profession"],
                "student_skills": row["skills"],
                "student_profile_text": profile_text,
            }
        )
    return students


def print_skill_lines(skills):
    if not skills:
        print("- None identified")
        return
    for skill in skills.split("; "):
        print(f"- {skill}")


def main():
    CV_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs()
    if TRAIN_JOBS_PATH.exists():
        train_jobs = pd.read_csv(TRAIN_JOBS_PATH)
    else:
        train_jobs, _ = split_jobs(jobs)
    vectorizer = fit_tfidf(train_jobs)
    skill_vocabulary = build_skill_vocabulary(jobs)

    students = choose_students()
    if not students:
        return

    job_description = input("Enter job description: ").strip()
    if not job_description:
        print("A job description is required.")
        return

    all_results = []
    for student in students:
        match = match_job_description(
            student["student_profile_text"],
            job_description,
            vectorizer,
            skill_vocabulary,
        )
        print_result(student, match)
        all_results.append({**student, "job_description": job_description, **match})

    output_path = RESULTS_DIR / "baseline_results.csv"
    save_results(all_results, output_path)
    print(f"\nResults saved to: {output_path}")


def print_result(student, result):
    print("\n## Student Profile")
    print(f"Identifier: {student['student_identifier']}")
    print(f"Degree: {student['degree'] or 'Not provided'}")
    print(f"Profession: {student['profession'] or 'Not provided'}")
    print(f"Skills: {student['student_skills'] or 'Extracted from profile text'}")

    print("\n## Job Description Match")
    print(f"Match Score: {result['match_score']:.2f}%")
    print("\nMatched Skills:")
    print_skill_lines(result["matched_skills"])
    print("\nMissing/Weak Skills:")
    print_skill_lines(result["missing_weak_skills"])
    print("\nRecommended Learning:")
    print_skill_lines(result["recommended_learning"])


if __name__ == "__main__":
    main()
