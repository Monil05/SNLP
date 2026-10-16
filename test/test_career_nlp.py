"""Tests for the Week 6 reusable asset (career_nlp).

Fast and offline: they use a small synthetic job dataset and either the LSA
fallback or a deterministic stub embedder, so no model download is needed.
Run:  pytest "base version/test/test_career_nlp.py"
"""
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from career_nlp import (  # noqa: E402
    AssetConfig, CareerNLP, CVProcessingError, InputValidationError, ModelLoadError,
)
from career_nlp.embedders import SentenceTransformerEmbedder, find_local_snapshot  # noqa: E402
from career_nlp.extraction import extract_degrees, extract_phones, skill_confidence  # noqa: E402
from career_nlp.text import clean_job_title, parse_skills  # noqa: E402

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
_TEMPLATES = {
    "INFORMATION-TECHNOLOGY": (["python", "sql", "machine learning", "data analysis", "power bi", "aws"],
                               ["software engineer", "data analyst", "systems administrator"]),
    "FINANCE": (["financial reporting", "budgeting", "forecasting", "excel", "accounting", "sap"],
                ["finance analyst", "accountant", "financial controller"]),
    "HR": (["recruitment", "onboarding", "payroll", "employee relations", "talent acquisition", "hris"],
           ["hr generalist", "recruiter", "hr manager"]),
    "SALES": (["lead generation", "negotiation", "crm", "cold calling", "account management", "forecasting"],
              ["sales representative", "account executive", "sales manager"]),
}


def _make_jobs(per_category: int = 12) -> pd.DataFrame:
    rows = []
    for cat, (skills, titles) in _TEMPLATES.items():
        for i in range(per_category):
            chosen = [skills[(i + k) % len(skills)] for k in range(4)]
            title = titles[i % len(titles)]
            if i % 5 == 0:
                title += " II"
            rows.append({
                "job_id": f"job_{cat[:3]}_{i:03d}", "category": cat, "job_title": title,
                "job_description": f"We are hiring a {title}. Responsibilities involve {', '.join(chosen)} "
                                   f"and reporting to the {cat.lower().replace('-', ' ')} team. Requires {chosen[0]}.",
                "job_skill_set": str(chosen),
            })
    return pd.DataFrame(rows)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder (no dependencies)."""
    name, dimensions = "hash-64", 64

    def encode(self, texts):
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, sum(map(ord, w)) % self.dimensions] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True); norms[norms == 0] = 1
        return out / norms


@pytest.fixture(scope="module")
def jobs():
    return _make_jobs()


@pytest.fixture(scope="module")
def nlp(jobs):
    return CareerNLP(AssetConfig(embedding_backend="lsa", lsa_dimensions=16), jobs=jobs).warmup()


IT_STUDENT = "B.Tech in Computer Science. Skills: python, sql, machine learning, data analysis. 2 years experience."
IT_JD = "Looking for a data analyst skilled in python, sql, power bi and aws."

# --------------------------------------------------------------------------
# structured output + confidence
# --------------------------------------------------------------------------
def test_lazy_init(jobs):
    n = CareerNLP(AssetConfig(embedding_backend="lsa", lsa_dimensions=16), jobs=jobs)
    assert not n.is_ready
    n.extract("python")
    assert n.is_ready


def test_match_structure_and_default_hides_internals(nlp):
    r = nlp.match(IT_STUDENT, IT_JD)
    d = json.loads(r.model_dump_json(exclude_none=True))
    assert 0 <= d["match_score"] <= 100
    assert {"python", "sql"} <= set(d["matched_skills"])
    assert {"power bi", "aws"} <= set(d["missing_weak_skills"])
    assert set(d["recommended_learning"]) == set(d["missing_weak_skills"])
    assert d["confidence"]["level"] in {"low", "medium", "high"} and d["confidence"]["calibrated"] is False
    assert "score_breakdown" not in d and "student_profile" not in d
    assert "factors" not in d["confidence"]
    assert d["meta"]["schema_version"] == "1.0" and d["meta"]["processing_ms"] >= 0


def test_match_details_opt_in(nlp):
    r = nlp.match(IT_STUDENT, IT_JD, include_details=True)
    b = r.score_breakdown
    assert b.skill_weight == 0.7 and b.semantic_weight == 0.3
    assert r.match_score == pytest.approx(0.7 * b.skill_match_percent + 0.3 * b.semantic_similarity_percent, abs=0.02)
    assert r.confidence.factors and r.student_profile.skills


def test_match_score_formula_and_extremes(nlp):
    full = nlp.match("python sql power bi aws machine learning", "python sql power bi aws", include_details=True)
    assert full.score_breakdown.skill_match_percent == 100.0
    none = nlp.match("recruitment payroll onboarding", "python sql power bi aws", include_details=True)
    assert none.score_breakdown.skill_match_percent == 0.0
    assert full.match_score > none.match_score


def test_match_warns_when_jd_has_no_known_skills(nlp):
    r = nlp.match(IT_STUDENT, "We want a friendly person who enjoys cake.")
    assert any("No known skills were recognised in the job description" in w for w in r.meta.warnings)
    assert r.confidence.level == "low"


def test_learning_order_is_by_dataset_demand(nlp):
    r = nlp.match("recruitment", "python sql power bi aws machine learning data analysis")
    st = nlp._state
    demand = [st.skill_demand[s] for s in r.recommended_learning]
    assert demand == sorted(demand, reverse=True)


def test_recommend_structure(nlp):
    r = nlp.recommend(IT_STUDENT, top_k=3, include_details=True)
    assert r.recommended_job.rank == 1 and r.recommended_job.category == "INFORMATION-TECHNOLOGY"
    assert len(r.alternatives) == 2
    titles = [r.recommended_job.job_title] + [a.job_title for a in r.alternatives]
    assert len(set(t.lower() for t in titles)) == 3           # alternatives are title-distinct
    assert all(not t.endswith(" II") for t in titles)          # level suffix cleaned
    assert set(r.confidence.factors) == {"category_agreement_top5", "top_job_skill_coverage",
                                         "profile_skill_evidence", "score_margin_to_second"}
    assert r.student_profile is not None


def test_recommend_no_skills_warns_and_is_low_confidence(nlp):
    r = nlp.recommend("I like long walks and good coffee")
    assert r.confidence.level == "low"
    assert any("No known skills" in w for w in r.meta.warnings)


def test_recommend_category_split_caps_confidence():
    from career_nlp import confidence as c
    split = c.recommend_confidence(80, 40, 100, category_agreement=0.4, n_student_skills=9)
    solid = c.recommend_confidence(80, 40, 100, category_agreement=1.0, n_student_skills=9)
    assert split.level == "low" and split.score < c.MEDIUM
    assert solid.level == "high"


def test_classify_confidence(nlp):
    r = nlp.classify("finance analyst: budgeting forecasting accounting excel sap financial reporting")
    assert r.predicted_category == "FINANCE"
    assert r.confidence.score == r.top_categories[0].probability
    assert r.confidence.method == "classifier_probability"
    assert sum(c.probability for c in nlp.classify("python sql", top_k=5).top_categories) == pytest.approx(1.0, abs=1e-3)


def test_analyze_dispatches(nlp):
    assert nlp.analyze(IT_STUDENT).flow == "jd_not_provided"
    assert nlp.analyze(IT_STUDENT, "  ").flow == "jd_not_provided"
    a = nlp.analyze(IT_STUDENT, IT_JD)
    assert a.flow == "jd_provided" and a.match and a.recommendation is None


def test_retrieve_embed_similarity_summarize(nlp):
    r = nlp.retrieve("python sql machine learning", top_k=3)
    assert [x.rank for x in r.results] == [1, 2, 3]
    assert r.results[0].similarity >= r.results[1].similarity >= r.results[2].similarity
    assert "Job " in r.context
    e = nlp.embed(["a python developer", "an hr recruiter"])
    assert len(e.vectors) == 2 and e.dimensions == len(e.vectors[0])
    s = nlp.similarity("python sql", "python sql")
    assert s.similarity == pytest.approx(1.0, abs=1e-3) and s.similarity_percent == pytest.approx(100, abs=0.1)
    text = "Python is used for data. SQL queries tables. Dashboards use power bi. Cats nap. Lunch is nice."
    sm = nlp.summarize(text, max_sentences=2)
    assert sm.sentences_total == 5 and sm.sentences_used == 2
    assert [x.index for x in sm.selected] == sorted(x.index for x in sm.selected)


def test_week3_compat_shape(nlp):
    d = CareerNLP.to_week3_dict(nlp.match(IT_STUDENT, IT_JD))
    assert set(d) == {"match_score", "matched_skills", "missing_weak_skills", "recommended_learning"}
    assert isinstance(d["matched_skills"], str) and "; " in d["matched_skills"]
    r = CareerNLP.to_week3_dict(nlp.recommend(IT_STUDENT))
    assert set(r) == {"recommended_job", "match_score", "matched_skills", "missing_weak_skills", "recommended_learning"}


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("I want to be a data analyst, contact me anytime", []),                  # Week 3 gave B.E. + M.E.
    ("Proficient in MS Office and Excel", []),                                # Week 3 gave M.S. in Office
    ("B.Tech Data Science with 3 years experience", ["B.Tech in Data Science"]),
    ("btech in computer science", ["B.Tech in Computer Science"]),
    ("M.S. in Computer Science", ["M.S. in Computer Science"]),
    ("BE in Electronics", ["B.E. in Electronics"]),
    ("MBA (Finance)", ["MBA in Finance"]),
    ("CFA level 2", ["CFA"]),
])
def test_degree_extraction(text, expected):
    assert [d.text for d in extract_degrees(text)] == expected


def test_phone_rejects_year_runs():
    assert extract_phones("worked 2019 2020 2021 and 2015-2016-2017") == []
    got = extract_phones("call +91 98765 43210 or 9876543210")
    assert len(got) == 2 and got[0].confidence > got[1].confidence - 0.2


def test_extract_response(nlp):
    r = nlp.extract("Jane, a@b.com, +91 98765 43210. B.Tech in Data Science, python, sql. Python again. 3 years experience.")
    names = {s.name: s for s in r.skills}
    assert names["python"].mentions == 2 and names["python"].confidence >= 0.9
    assert r.emails[0].value == "a@b.com" and r.experience_years == 3.0
    assert r.entities_count == len(r.skills) + len(r.degrees_or_certifications) + len(r.emails) + len(r.phone_numbers) + len(r.experience_mentions)


def test_skill_confidence_rules():
    assert skill_confidence("r", 1) < skill_confidence("sap", 1) < skill_confidence("power bi", 1)
    assert skill_confidence("python", 2) > skill_confidence("python", 1)


def test_text_helpers():
    assert clean_job_title("data solutions analyst ii (managed care - finance focus) - r7269") == "data solutions analyst"
    assert parse_skills("['Power  BI', 'SQL', 'sql']") == ["power bi", "sql"]
    assert parse_skills("a; b") == ["a", "b"]


# --------------------------------------------------------------------------
# validation + errors
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_rejects_bad_text(nlp, bad):
    with pytest.raises(InputValidationError):
        nlp.extract(bad)


def test_rejects_oversized_and_bad_topk(jobs):
    small = CareerNLP(AssetConfig(embedding_backend="lsa", lsa_dimensions=16, max_text_chars=20), jobs=jobs)
    with pytest.raises(InputValidationError) as ei:
        small.extract("x" * 21)
    assert ei.value.details["max"] == 20
    with pytest.raises(InputValidationError):
        small.recommend("python", top_k=0)


def test_bad_dataset_errors(tmp_path):
    with pytest.raises(ModelLoadError):
        CareerNLP(AssetConfig(jobs_path=tmp_path / "nope.csv", embedding_backend="lsa")).warmup()
    with pytest.raises(ModelLoadError) as ei:
        CareerNLP(AssetConfig(embedding_backend="lsa"), jobs=pd.DataFrame({"job_id": ["a"]})).warmup()
    assert "category" in ei.value.details["missing"]


def test_config_validation():
    with pytest.raises(ValueError):
        AssetConfig(skill_weight=0.5, semantic_weight=0.6)
    with pytest.raises(ValueError):
        AssetConfig(embedding_backend="bogus")


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CAREER_NLP_JOBS_PATH", str(tmp_path / "j.csv"))
    monkeypatch.setenv("CAREER_NLP_BACKEND", "lsa")
    monkeypatch.setenv("CAREER_NLP_WARMUP", "false")
    c = AssetConfig.from_env()
    assert c.jobs_path == tmp_path / "j.csv" and c.embedding_backend == "lsa" and c.warmup_on_start is False


# --------------------------------------------------------------------------
# pluggable embedder, cache, Sentence-Transformer loader (stubbed)
# --------------------------------------------------------------------------
def test_custom_embedder_and_disk_cache(jobs, tmp_path):
    cfg = AssetConfig(cache_dir=tmp_path / "cache")
    a = CareerNLP(cfg, jobs=jobs, embedder=HashEmbedder()).warmup()
    files = list((tmp_path / "cache").glob("job_index_*.npz"))
    assert len(files) == 1
    mtime = files[0].stat().st_mtime_ns
    b = CareerNLP(cfg, jobs=jobs, embedder=HashEmbedder()).warmup()   # reuses cache
    assert files[0].stat().st_mtime_ns == mtime
    assert np.allclose(a._state.index_vectors, b._state.index_vectors)
    assert a.match(IT_STUDENT, IT_JD).meta.model.embedding_backend == "custom"
    # changing the data invalidates the cache key
    CareerNLP(cfg, jobs=jobs.iloc[:-1], embedder=HashEmbedder()).warmup()
    assert len(list((tmp_path / "cache").glob("job_index_*.npz"))) == 2


def test_auto_backend_falls_back_with_warning(jobs, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)   # force ImportError
    n = CareerNLP(AssetConfig(embedding_backend="auto", lsa_dimensions=16), jobs=jobs)
    r = n.extract("python")
    assert r.meta.model.embedding_backend == "lsa"
    assert any("LSA offline fallback" in w for w in r.meta.warnings)
    strict = CareerNLP(AssetConfig(embedding_backend="sentence-transformers"), jobs=jobs)
    with pytest.raises(ModelLoadError):
        strict.warmup()


def test_sentence_transformer_loader_uses_local_snapshot(tmp_path, monkeypatch):
    snap = tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}"); (snap / "model.safetensors").write_text("x")
    assert find_local_snapshot("all-MiniLM-L6-v2", tmp_path) == snap
    assert find_local_snapshot("other-model", tmp_path) is None

    calls = []

    class FakeST:
        def __init__(self, path, **kw):
            calls.append((path, kw))
        def get_sentence_embedding_dimension(self):
            return 384
        def encode(self, texts, **kw):
            return np.ones((len(texts), 384), dtype=np.float32) / np.sqrt(384)

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2", tmp_path)
    assert calls == [(str(snap), {})]                 # loaded from disk, no download args
    assert emb.dimensions == 384 and emb.encode(["a", "b"]).shape == (2, 384)
    # no snapshot -> hub name + cache_folder (download path)
    SentenceTransformerEmbedder("all-MiniLM-L6-v2", tmp_path / "empty")
    assert calls[-1][0] == "all-MiniLM-L6-v2" and "cache_folder" in calls[-1][1]


# --------------------------------------------------------------------------
# CV
# --------------------------------------------------------------------------
def _docx_bytes(paragraphs, table_cells=()):
    from docx import Document
    d = Document()
    for p in paragraphs:
        d.add_paragraph(p)
    if table_cells:
        t = d.add_table(rows=1, cols=len(table_cells))
        for c, txt in zip(t.rows[0].cells, table_cells):
            c.text = txt
    buf = io.BytesIO(); d.save(buf)
    return buf.getvalue()


def test_cv_docx_including_tables(nlp):
    data = _docx_bytes(["Jane Doe  jane@x.com", "B.Tech in Data Science. 2 years experience."],
                       table_cells=["Skills", "python, sql, machine learning"])
    r = nlp.analyze_cv_bytes(data, "jane.docx")
    assert r.filename == "jane.docx" and {"python", "sql"} <= {s.name for s in r.entities.skills}
    assert r.profile.normalized_education == "B.Tech in Data Science"


def test_cv_errors(nlp):
    with pytest.raises(CVProcessingError):
        nlp.analyze_cv_bytes(b"hello", "cv.txt")
    with pytest.raises(CVProcessingError):
        nlp.analyze_cv_bytes(b"not really a pdf", "cv.pdf")
    with pytest.raises(CVProcessingError):
        nlp.analyze_cv_bytes(_docx_bytes([""]), "empty.docx")
    with pytest.raises(CVProcessingError):
        nlp.analyze_cv("/no/such/file.pdf")


# --------------------------------------------------------------------------
# REST API
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(nlp):
    from fastapi.testclient import TestClient
    from career_nlp.api import create_app
    return TestClient(create_app(engine=nlp))


def test_api_health_info(client):
    h = client.get("/v1/health").json()
    assert h["status"] == "ok" and h["ready"] and h["jobs_indexed"] == 48
    i = client.get("/v1/info").json()
    assert i["scoring"] == {"skill_weight": 0.7, "semantic_weight": 0.3} and "HR" in i["categories"]


def test_api_match_and_recommend(client):
    r = client.post("/v1/match", json={"student_text": IT_STUDENT, "job_text": IT_JD})
    assert r.status_code == 200 and r.headers["X-Request-ID"]
    body = r.json()
    assert "score_breakdown" not in body and body["confidence"]["level"]
    d = client.post("/v1/match", json={"student_text": IT_STUDENT, "job_text": IT_JD, "include_details": True}).json()
    assert d["score_breakdown"]["skill_weight"] == 0.7
    rec = client.post("/v1/recommend", json={"student_text": IT_STUDENT, "top_k": 2}).json()
    assert rec["recommended_job"]["category"] == "INFORMATION-TECHNOLOGY" and len(rec["alternatives"]) == 1
    both = client.post("/v1/analyze", json={"student_text": IT_STUDENT}).json()
    assert both["flow"] == "jd_not_provided" and "match" not in both


@pytest.mark.parametrize("path,payload", [
    ("/v1/extract", {"text": IT_STUDENT}), ("/v1/classify", {"text": IT_STUDENT, "top_k": 2}),
    ("/v1/embed", {"texts": ["a", "b"]}), ("/v1/similarity", {"text_a": "a b", "text_b": "a b"}),
    ("/v1/summarize", {"text": "One. Two. Three. Four."}), ("/v1/retrieve", {"query": "python", "top_k": 2}),
])
def test_api_nlp_endpoints(client, path, payload):
    r = client.post(path, json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["schema_version"] == "1.0"


def test_api_error_contract(client):
    r = client.post("/v1/match", json={"student_text": "", "job_text": "x"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
    assert r.json()["error"]["request_id"] == r.headers["X-Request-ID"]
    r = client.post("/v1/match", json={"student_text": "   ", "job_text": "x"})   # passes pydantic, fails engine
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"
    r = client.post("/v1/match", json={"student_text": "a", "job_text": "b", "bogus": 1})
    assert r.status_code == 422                                                   # extra fields forbidden
    r = client.post("/v1/recommend", json={"student_text": "a", "top_k": 99})
    assert r.status_code == 422
    assert client.get("/v1/nope").status_code == 404


def test_api_cv_upload(client):
    data = _docx_bytes(["B.Tech in Data Science", "python sql"])
    r = client.post("/v1/cv/analyze", files={"file": ("cv.docx", data, "application/octet-stream")})
    assert r.status_code == 200 and r.json()["profile"]["skills"]
    r = client.post("/v1/cv/analyze", files={"file": ("cv.exe", b"x", "application/octet-stream")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "cv_processing_failed"


def test_api_key_enforced(nlp, monkeypatch):
    from fastapi.testclient import TestClient
    from career_nlp.api import create_app
    monkeypatch.setenv("CAREER_NLP_API_KEY", "s3cret")
    c = TestClient(create_app(engine=nlp))
    assert c.get("/v1/health").status_code == 200                                  # health stays open
    assert c.get("/v1/info").status_code == 401
    assert c.get("/v1/info", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/v1/info", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_api_health_degraded_before_warmup(jobs):
    from fastapi.testclient import TestClient
    from career_nlp.api import create_app
    cold = CareerNLP(AssetConfig(embedding_backend="lsa", lsa_dimensions=16, warmup_on_start=False), jobs=jobs)
    c = TestClient(create_app(engine=cold))
    h = c.get("/v1/health").json()
    assert h["status"] == "degraded" and h["ready"] is False
    assert c.post("/v1/extract", json={"text": "python"}).status_code == 200      # lazy-inits on demand
    assert c.get("/v1/health").json()["ready"] is True


def test_cli_roundtrip(jobs, tmp_path, capsys):
    from career_nlp.cli import main
    p = tmp_path / "jobs.csv"; jobs.to_csv(p, index=False)
    assert main(["--backend", "lsa", "--jobs", str(p), "extract", "--text", "python and sql"]) == 0
    assert "python" in json.loads(capsys.readouterr().out)["skills"][0]["name"]
    assert main(["--backend", "lsa", "--jobs", str(p), "match", "--student", "", "--job", "x"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_input"
