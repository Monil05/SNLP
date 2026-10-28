import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src" / "week3"
sys.path.insert(0, str(SRC))

from nlp_v1 import V1NLP


def make_pipeline():
    return V1NLP()


def test_extract_entities():
    p = make_pipeline()
    result = p.extract_entities("B.Tech Data Science. Python, SQL and Power BI. 3 years experience.")
    assert "python" in result["skills"]
    assert "sql" in result["skills"]
    assert result["degrees_or_certifications"]
    assert result["experience_mentions"]


def test_similarity_range():
    p = make_pipeline()
    result = p.similarity("python data analysis", "python data analysis")
    assert 0.0 <= result["similarity"] <= 1.0


def test_classification():
    p = make_pipeline()
    result = p.classify_job("finance accounting budgeting forecasting Excel SAP")
    assert result["predicted_category"]
    assert result["top_categories"]


def test_rag():
    p = make_pipeline()
    result = p.rag("Python data analysis machine learning", top_k=3)
    assert len(result["retrieved_jobs"]) == 3


def test_full_pipeline():
    p = make_pipeline()
    result = p.pipeline("B.Tech Python SQL machine learning", "Python SQL machine learning developer", top_k=2)
    assert result["version"] == "1.0"
    assert "match" in result
    assert len(result["rag_retrieval"]) == 2
