from pathlib import Path

import pandas as pd

from baseline import clean_description, clean_title, format_skills, parse_skills, split_jobs


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "dataset" / "job_skill_set.parquet"
DATA_DIR = BASE_DIR / "data"


TEST_STUDENTS = [
    {"student_id": "student_01", "degree": "B.Tech", "profession": "Data Science Student", "skills": "python, sql, machine learning, data analysis, power bi, communication"},
    {"student_id": "student_02", "degree": "BBA", "profession": "HR Intern", "skills": "recruitment, employee relations, onboarding, payroll processing, communication"},
    {"student_id": "student_03", "degree": "MBA", "profession": "Finance Analyst", "skills": "financial analysis, budgeting, forecasting, excel, variance analysis, reporting"},
    {"student_id": "student_04", "degree": "B.Com", "profession": "Sales Associate", "skills": "customer service, sales, negotiation, cold calling, crm, relationship building"},
    {"student_id": "student_05", "degree": "B.Tech", "profession": "IT Support Student", "skills": "network administration, troubleshooting, microsoft windows, technical support, communication"},
    {"student_id": "student_06", "degree": "MBA", "profession": "Business Development Student", "skills": "business development, market analysis, lead generation, presentation skills, negotiation"},
    {"student_id": "student_07", "degree": "B.Tech", "profession": "Software Developer", "skills": "programming, database concepts, api development, problem solving, collaboration"},
    {"student_id": "student_08", "degree": "M.Com", "profession": "Accounting Student", "skills": "accounting, financial reporting, accounts payable, excel, attention to detail"},
    {"student_id": "student_09", "degree": "BBA", "profession": "Marketing and Sales Student", "skills": "presentation skills, customer relationship management, sales, communication, teamwork"},
    {"student_id": "student_10", "degree": "MBA", "profession": "Operations HR Student", "skills": "hr policies, training and development, performance management, compliance, confidentiality"},
    {"student_id": "student_11", "degree": "B.Tech", "profession": "Cybersecurity Student", "skills": "it security, risk analysis, networking technologies, problem solving, documentation"},
    {"student_id": "student_12", "degree": "MBA", "profession": "Client Success Associate", "skills": "account management, relationship building, customer service, communication, problem solving"},
    {"student_id": "student_13", "degree": "B.Com", "profession": "Financial Planning Student", "skills": "financial planning, fp&a, financial analysis, tableau, power bi, strategic planning"},
    {"student_id": "student_14", "degree": "B.Tech", "profession": "Cloud and Systems Student", "skills": "cloud tools, system design, disaster recovery planning, project management, collaboration"},
    {"student_id": "student_15", "degree": "BBA", "profession": "Recruitment Coordinator", "skills": "talent acquisition, background checks, employee onboarding, microsoft office, organizational skills"},
    {"student_id": "student_16", "degree": "MBA", "profession": "Strategic Sales Student", "skills": "strategic selling, territory management, crm, forecasting, networking, persuasion"},
]


def clean_jobs():
    df = pd.read_parquet(DATASET_PATH)
    cleaned = pd.DataFrame()
    cleaned["job_id"] = df["job_id"]
    cleaned["category"] = df["category"].fillna("").astype(str).str.strip()
    cleaned["job_title"] = df["job_title"].apply(clean_title)
    cleaned["job_description"] = df["job_description"].apply(clean_description)
    cleaned["job_skill_set"] = df["job_skill_set"].apply(lambda value: format_skills(parse_skills(value)))
    return cleaned


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = clean_jobs()
    cleaned_path = DATA_DIR / "cleaned_jobs.csv"
    cleaned.to_csv(cleaned_path, index=False)

    train_jobs, test_jobs = split_jobs(cleaned)
    train_jobs.to_csv(DATA_DIR / "train_jobs.csv", index=False)
    test_jobs.to_csv(DATA_DIR / "test_jobs.csv", index=False)

    students_path = BASE_DIR / "test" / "test_students.csv"
    students_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(TEST_STUDENTS).to_csv(students_path, index=False)

    print(f"Cleaned jobs saved to: {cleaned_path}")
    print(f"Cleaned job columns: {list(cleaned.columns)}")
    print(f"Train jobs: {len(train_jobs)}")
    print(f"Test jobs: {len(test_jobs)}")
    print(f"Test student inputs saved to: {students_path}")


if __name__ == "__main__":
    main()
