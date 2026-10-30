"""Minimal usage of the asset.  Run from 'base version/':
    python examples/quickstart.py [--backend lsa]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from career_nlp import AssetConfig, CareerNLP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="sentence-transformers", choices=["sentence-transformers", "lsa", "auto"])
nlp = CareerNLP(AssetConfig(embedding_backend=ap.parse_args().backend))

student = ("B.Tech in Data Science. Skills: python, sql, machine learning, data analysis, power bi, "
           "communication. 2 years experience.")
jd = "Data analyst needed: python, sql, power bi, tableau and strong communication skills."

m = nlp.match(student, jd)                       # JD provided
print(f"Match {m.match_score}% | confidence {m.confidence.level} ({m.confidence.score})")
print("  matched:", m.matched_skills, "\n  learn next:", m.recommended_learning)

r = nlp.recommend(student, top_k=3)              # JD not provided
print(f"\nBest job: {r.recommended_job.job_title} [{r.recommended_job.category}] "
      f"{r.recommended_job.match_score}% | confidence {r.confidence.level}")
for alt in r.alternatives:
    print(f"  alt: {alt.job_title} {alt.match_score}%")
for w in r.meta.warnings:
    print("  warning:", w)
