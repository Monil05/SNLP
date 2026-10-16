"""Re-check the recommend-confidence heuristic against the 16 labelled synthetic
students (expected categories from week4/evaluation_week4.py).

    python examples/validate_confidence.py [--backend lsa|sentence-transformers]

Prints per-student results and top-1-category accuracy by confidence level.
Use this after changing backend, data or thresholds - the numbers in
WEEK6_ASSET_ENGINEERING.md were produced with --backend lsa.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from career_nlp import AssetConfig, CareerNLP  # noqa: E402

EXPECTED = {"student_01": "INFORMATION-TECHNOLOGY", "student_02": "HR", "student_03": "FINANCE",
            "student_04": "SALES", "student_05": "INFORMATION-TECHNOLOGY", "student_06": "BUSINESS-DEVELOPMENT",
            "student_07": "INFORMATION-TECHNOLOGY", "student_08": "FINANCE", "student_09": "SALES",
            "student_10": "HR", "student_11": "INFORMATION-TECHNOLOGY", "student_12": "SALES",
            "student_13": "FINANCE", "student_14": "INFORMATION-TECHNOLOGY", "student_15": "HR",
            "student_16": "SALES"}

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="sentence-transformers", choices=["sentence-transformers", "lsa", "auto"])
nlp = CareerNLP(AssetConfig(embedding_backend=ap.parse_args().backend)).warmup()

rows = []
for _, s in pd.read_csv(ROOT / "test" / "test_students.csv").iterrows():
    text = " ".join([str(s.degree), str(s.profession), str(s.skills)])
    r = nlp.recommend(text, include_details=True)
    rows.append({"student": s.student_id, "expected": EXPECTED[s.student_id],
                 "got": r.recommended_job.category, "correct": r.recommended_job.category == EXPECTED[s.student_id],
                 "confidence": r.confidence.score, "level": r.confidence.level,
                 "category_agreement": r.confidence.factors["category_agreement_top5"]})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\ntop-1 category accuracy: {df.correct.mean():.3f}  (n={len(df)})")
print(df.groupby("level").correct.agg(["count", "mean"]).rename(columns={"mean": "accuracy"}))
print("mean confidence  correct: %.3f   wrong: %.3f" % (df[df.correct].confidence.mean(),
                                                         df[~df.correct].confidence.mean()))
