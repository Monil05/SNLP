"""Small command-line demo for the Week 3 core NLP pipeline.

Demonstrates both required user flows:
  1. JD PROVIDED    -> match_with_jd()
  2. JD NOT PROVIDED -> recommend_job()
"""
from pathlib import Path
import sys

WEEK3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEEK3_DIR))
from nlp_v1 import V1NLP


def print_skill_lines(skills: str) -> None:
    if not skills:
        print("- None identified")
        return
    for skill in skills.split("; "):
        print(f"- {skill}")


def main():
    nlp = V1NLP()
    student = input("Student/CV text: ").strip()
    job = input("Job description (leave blank for a job recommendation instead): ").strip()

    print("\n=== WEEK 3 CORE NLP PIPELINE ===")
    profile = nlp.derive_student_profile(student)
    print("\nDerived student profile:")
    print(f"- Normalized education: {profile['normalized_education']}")
    print(f"- Skills: {'; '.join(profile['skills']) or 'None identified'}")
    print(f"- Experience: {profile['experience']}")

    if job:
        result = nlp.match_with_jd(student, job)
        print("\n--- JD Provided: Job Description Match ---")
        print(f"Match Score: {result['match_score']}%")
        print("\nMatched Skills:")
        print_skill_lines(result["matched_skills"])
        print("\nMissing/Weak Skills:")
        print_skill_lines(result["missing_weak_skills"])
        print("\nRecommended Learning:")
        print_skill_lines(result["recommended_learning"])
    else:
        result = nlp.recommend_job(student)
        print("\n--- JD Not Provided: Recommended Job ---")
        print(f"Recommended Job: {result['recommended_job']}")
        print(f"Match Score: {result['match_score']}%")
        print("\nMatched Skills:")
        print_skill_lines(result["matched_skills"])
        print("\nMissing/Weak Skills:")
        print_skill_lines(result["missing_weak_skills"])
        print("\nRecommended Learning:")
        print_skill_lines(result["recommended_learning"])


if __name__ == "__main__":
    main()
