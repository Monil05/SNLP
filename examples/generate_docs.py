"""Regenerate docs/openapi.json, docs/schemas/*.json and docs/sample_outputs/*.json.
    python examples/generate_docs.py [--backend lsa]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from career_nlp import AssetConfig, CareerNLP, schemas  # noqa: E402
from career_nlp.api import create_app  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="sentence-transformers", choices=["sentence-transformers", "lsa", "auto"])
backend = ap.parse_args().backend
docs = ROOT / "docs"
(docs / "schemas").mkdir(parents=True, exist_ok=True)
(docs / "sample_outputs").mkdir(parents=True, exist_ok=True)

nlp = CareerNLP(AssetConfig(embedding_backend=backend)).warmup()
(docs / "openapi.json").write_text(json.dumps(create_app(engine=nlp).openapi(), indent=2))

for name in ["AcademicRecord", "AcademicAdjustment", "CompetencyRequirement", "ExtractResponse", "ClassifyResponse",
             "MatchRequest", "MatchResponse", "RecommendRequest", "RecommendResponse",
             "CompareJobsRequest", "CompareJobsResponse", "JobDescriptionInput", "ComparedJob",
             "RetrieveResponse", "SummarizeResponse", "CVResponse", "ErrorResponse"]:
    (docs / "schemas" / f"{name}.schema.json").write_text(json.dumps(getattr(schemas, name).model_json_schema(), indent=2))

student = ("B.Tech in Data Science. Skills: python, sql, machine learning, data analysis, power bi, "
           "communication. 2 years experience.")
jd = "Data analyst needed: python, sql, power bi, tableau and strong communication skills."
samples = {
    "extract": nlp.extract(student + " Physics: 35%. Contact: jane@example.com, +91 98765 43210."),
    "classify": nlp.classify("finance analyst budgeting forecasting accounting excel sap"),
    "match": nlp.match(student, jd),
    "match_details": nlp.match(student, jd, include_details=True),
    "recommend": nlp.recommend(student, top_k=2),
    "recommend_details": nlp.recommend(student, top_k=1, include_details=True),
    "compare_jobs": nlp.compare_jobs(
        "B.Tech Data Science. Python, SQL. Physics: 35%.",
        [
            {"job_id": "JD1", "job_title": "Physics-focused analyst",
             "job_text": "Data analyst requiring strong expertise in Physics. Python and SQL."},
            {"job_id": "JD2", "job_title": "General data analyst",
             "job_text": "Data analyst using Python and SQL. Physics expertise is not required."},
        ],
    ),
}
for key, val in samples.items():
    (docs / "sample_outputs" / f"{key}.json").write_text(val.model_dump_json(indent=2, exclude_none=True))
print("wrote docs for backend:", backend)
