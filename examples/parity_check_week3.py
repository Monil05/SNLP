"""Compare the asset against the original Week 3 V1NLP on the same inputs.

Run this on a machine where sentence-transformers is installed (it was NOT run
during development - the build sandbox could not install PyTorch):

    python examples/parity_check_week3.py

With the Sentence-Transformer backend, match scores should agree to within
rounding for texts that don't trigger the Week 3 degree-regex false positives
(e.g. 'to be', 'contact me', 'MS Office'); the script reports every difference.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "week3"))
from career_nlp import AssetConfig, CareerNLP  # noqa: E402
from nlp_v1 import V1NLP  # noqa: E402

new, old = CareerNLP(AssetConfig()).warmup(), V1NLP()
students = pd.read_csv(ROOT / "test" / "test_students.csv")
jd = "Data analyst: python, sql, power bi, tableau, data analysis, communication."
worst = 0.0
for _, s in students.iterrows():
    text = " ".join([str(s.degree), str(s.profession), str(s.skills)])
    a, b = new.match(text, jd).match_score, old.match_with_jd(text, jd)["match_score"]
    worst = max(worst, abs(a - b))
    print(f"{s.student_id}: asset={a:6.2f} week3={b:6.2f} diff={a - b:+.2f}")
    ra, rb = new.recommend(text).recommended_job, old.recommend_job(text)
    if ra.job_title != rb["recommended_job"] or abs(ra.match_score - rb["match_score"]) > 0.05:
        print(f"   recommend differs: asset={ra.job_title!r} {ra.match_score} | week3={rb['recommended_job']!r} {rb['match_score']}")
print(f"max |match_score diff| = {worst:.2f}")
