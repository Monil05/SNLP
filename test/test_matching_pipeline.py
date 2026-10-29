"""Recommendation speed path, competency priority, and workflow regression tests."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "test"))

from career_nlp import AssetConfig, CareerNLP, InputValidationError  # noqa: E402
from career_nlp.text import parse_skills  # noqa: E402

from test_career_nlp import HashEmbedder, IT_JD, IT_STUDENT, _make_jobs  # noqa: E402


@pytest.fixture(scope="module")
def nlp():
    return _engine(_make_jobs())


def _engine(jobs, *, embedder=None, **config):
    return CareerNLP(
        AssetConfig(embedding_backend="lsa", lsa_dimensions=16, **config),
        jobs=jobs,
        embedder=embedder if embedder is not None else HashEmbedder(),
    ).warmup()


def test_negated_skill_is_not_demonstrated_or_matched(tmp_path):
    jobs = pd.DataFrame([
        {
            "job_id": "dev",
            "category": "INFORMATION-TECHNOLOGY",
            "job_title": "Developer",
            "job_description": "Programming experience is required. Python is required.",
            "job_skill_set": "['programming', 'python', 'communication']",
        },
        {
            "job_id": "other",
            "category": "FINANCE",
            "job_title": "Accountant",
            "job_description": "Financial reporting is required.",
            "job_skill_set": "['financial reporting']",
        },
    ])
    engine = _engine(jobs, cache_dir=tmp_path / "cache")
    negated = engine.derive_student_profile(
        "B.A. in English. Skills: communication, writing. No programming experience."
    )
    assert "programming" not in {skill.casefold() for skill in negated.skills}
    assert "communication" in {skill.casefold() for skill in negated.skills}

    positive = engine.derive_student_profile(
        "Programming experience with Python. Built course projects in Python."
    )
    assert "programming" in {skill.casefold() for skill in positive.skills}
    assert "python" in {skill.casefold() for skill in positive.skills}

    denied = engine.match(
        "Skills: communication. No programming experience.",
        "Programming experience is required.",
        include_details=True,
    )
    assert "programming" not in {skill.casefold() for skill in denied.matched_skills}
    assert any(item.name.casefold() == "programming" and item.status == "missing"
               for item in denied.competency_requirements)

    shown = engine.match(
        "Programming experience with Python.",
        "Programming experience is required. Python is required.",
        include_details=True,
    )
    matched = {skill.casefold() for skill in shown.matched_skills}
    assert "programming" in matched
    assert "python" in matched


def test_reporting_cycle_does_not_become_a_required_skill(tmp_path):
    jobs = pd.DataFrame([
        {
            "job_id": "intern",
            "category": "FINANCE",
            "job_title": "Investor relations intern",
            "job_description": (
                "Essential responsibilities include support for the quarterly reporting cycle "
                "and tracking of financial reporting cycles."
            ),
            "job_skill_set": "['reporting', 'excel']",
        },
        {
            "job_id": "analyst",
            "category": "INFORMATION-TECHNOLOGY",
            "job_title": "Data analyst",
            "job_description": "Experience with SQL is required.",
            "job_skill_set": "['sql']",
        },
    ])
    engine = _engine(jobs, cache_dir=tmp_path / "cache")
    result = engine.recommend("Skills: reporting and SQL.", top_k=2, include_details=True)
    intern = result.recommended_job if result.recommended_job.job_id == "intern" else result.alternatives[0]
    reporting = next(item for item in intern.competency_requirements if item.name.casefold() == "reporting")
    assert reporting.priority != "required"
    assert "reporting" not in {name.casefold() for name in intern.missing_weak_skills}


def test_metadata_skills_are_not_automatically_required(tmp_path):
    jobs = pd.DataFrame([
        {
            "job_id": "role",
            "category": "INFORMATION-TECHNOLOGY",
            "job_title": "Data engineer",
            "job_description": (
                "What you'll do: own data management and visualize data. "
                "Who you are: you have experience with ETL. You are proficient in SQL. "
                "Skills that will help you stand out: Python scripting, data warehouse experience, "
                "and cloud technologies. "
                "Our culture: collaboration is embedded and we value innovation and customer focus. "
                "Communication is not required."
            ),
            "job_skill_set": str([
                "etl", "sql", "python", "data warehousing", "cloud technologies",
                "collaboration", "innovation", "customer focus", "communication", "adaptability",
            ]),
        },
        {
            "job_id": "other",
            "category": "FINANCE",
            "job_title": "Accountant",
            "job_description": "Prepare financial reporting and budgeting.",
            "job_skill_set": "['financial reporting', 'budgeting']",
        },
    ])
    engine = _engine(jobs, cache_dir=tmp_path / "cache")
    result = engine.recommend(
        "B.Tech Data Science. Skills: Python, SQL. Projects: wrote SQL queries and Python scripts.",
        top_k=2,
        include_details=True,
    )
    job = result.recommended_job if result.recommended_job.job_id == "role" else result.alternatives[0]
    requirements = {item.name.casefold(): item for item in job.competency_requirements}

    assert requirements["sql"].priority == "required" and requirements["sql"].status == "matched"
    assert requirements["etl"].priority == "required" and requirements["etl"].status == "missing"
    assert requirements["python"].priority == "preferred" and requirements["python"].status == "matched"
    assert requirements["communication"].priority == "not_required"
    assert requirements["adaptability"].priority == "preferred"
    assert requirements["collaboration"].priority == "preferred"
    assert "communication" not in job.missing_weak_skills
    assert "collaboration" not in job.missing_weak_skills
    assert "etl" in job.missing_weak_skills
    assert {"sql", "python"} <= set(job.matched_skills)
    assert job.recommended_learning.index("etl") < job.recommended_learning.index("adaptability")
    # Missing ETL keeps the score modest, while unmatched soft skills cannot pull it to the old ~2/10 coverage.
    assert 35 <= job.score_breakdown.skill_match_percent <= 75
    assert job.match_score == pytest.approx(
        max(0.0, job.base_match_score - job.academic_penalty_percent),
        abs=0.02,
    )
    assert job.score_breakdown.required_skills_detected == sum(
        item.priority == "required" for item in job.competency_requirements
    )


def test_generic_tier_cannot_outweigh_one_essential_skill(tmp_path):
    generic = [
        "communication", "collaboration", "adaptability", "customer focus",
        "problem solving", "innovation", "teamwork", "leadership",
    ]
    jobs = pd.DataFrame([
        {
            "job_id": "tech",
            "category": "INFORMATION-TECHNOLOGY",
            "job_title": "Python developer",
            "job_description": "Proficient in Python is required.",
            "job_skill_set": str(["python", *generic]),
        },
        {
            "job_id": "finance",
            "category": "FINANCE",
            "job_title": "Accountant",
            "job_description": "Financial reporting is required.",
            "job_skill_set": "['financial reporting']",
        },
    ])
    engine = _engine(jobs, cache_dir=tmp_path / "cache")
    covered = engine.match("Skills: Python, communication, collaboration, teamwork.", "Proficient in Python is required.", include_details=True)
    # Direct JD has no metadata tail; use the stored job via recommend for the generic list.
    recommended = engine.recommend("Skills: Python.", top_k=2, include_details=True)
    tech = recommended.recommended_job if recommended.recommended_job.job_id == "tech" else recommended.alternatives[0]
    missing_required = engine.recommend(
        "Skills: communication, collaboration, adaptability, teamwork, leadership, innovation.",
        top_k=2,
        include_details=True,
    )
    weak = missing_required.recommended_job if missing_required.recommended_job.job_id == "tech" else next(
        job for job in missing_required.alternatives if job.job_id == "tech"
    )
    assert tech.score_breakdown.skill_match_percent > 80
    assert weak.score_breakdown.skill_match_percent < 25
    assert covered.score_breakdown.skill_match_percent == 100
    assert "python" not in {name.casefold() for name in tech.missing_weak_skills}


def test_aliases_match_paraphrases_without_duplicate_requirements(nlp):
    result = nlp.match(
        "Skills: SQL, Python, ETL pipelines. Projects: built data pipelines and relational database querying.",
        "Experience with relational database querying is required. "
        "Python development is required. Experience with data pipeline development is required. "
        "Experience with warehouse modelling is preferred.",
        include_details=True,
    )
    names = [item.name.casefold() for item in result.competency_requirements]
    assert names.count("sql") == 1
    assert "relational database querying" not in names
    assert {"sql", "python", "etl"} <= set(result.matched_skills)
    assert "data warehousing" not in result.missing_weak_skills
    warehouse = next(item for item in result.competency_requirements if "warehous" in item.name.casefold())
    assert warehouse.priority == "preferred"
    assert warehouse.status == "missing"


def test_shortlist_keeps_strong_skill_matches_and_limits_assessment(tmp_path):
    rows = []
    for index in range(30):
        rows.append({
            "job_id": f"noise_{index}",
            "category": "FINANCE" if index % 2 else "SALES",
            "job_title": f"Account executive {index}",
            "job_description": "Prospecting, negotiation, forecasting and account management for commercial clients.",
            "job_skill_set": "['negotiation', 'crm', 'forecasting']",
        })
    rows.append({
        "job_id": "hidden_skill",
        "category": "INFORMATION-TECHNOLOGY",
        "job_title": "Gallery cataloguer",
        "job_description": "The ancient marble statue rests in a quiet gallery of pottery and textiles.",
        "job_skill_set": "['python', 'sql', 'machine learning']",
    })
    rows.append({
        "job_id": "plain_it",
        "category": "INFORMATION-TECHNOLOGY",
        "job_title": "Software engineer",
        "job_description": "Build services with python, sql and machine learning.",
        "job_skill_set": "['python', 'sql', 'machine learning']",
    })
    jobs = pd.DataFrame(rows)
    engine = _engine(jobs, cache_dir=tmp_path / "cache", candidate_shortlist_size=8, retrieval_skill_slots=4)
    result = engine.recommend(IT_STUDENT, top_k=5, include_details=True)
    stats = engine.last_recommendation_stats
    assert stats["jobs_assessed"] <= 8
    assert stats["jobs_assessed"] < len(jobs)
    assert "hidden_skill" in stats["assessed_job_ids"]
    assert "plain_it" in stats["assessed_job_ids"]
    ranked = [result.recommended_job, *result.alternatives]
    assert ranked[0].category == "INFORMATION-TECHNOLOGY"
    assert {job.job_id for job in ranked} & {"hidden_skill", "plain_it"}


def test_full_ranking_reference_agrees_when_shortlist_covers_corpus():
    jobs = _make_jobs()
    engine = _engine(jobs, candidate_shortlist_size=200, retrieval_skill_slots=80)
    fast = engine.recommend(IT_STUDENT, top_k=5, include_details=True)
    full = engine.recommend(IT_STUDENT, top_k=5, include_details=True, full_ranking=True)
    fast_ids = [fast.recommended_job.job_id, *[job.job_id for job in fast.alternatives]]
    full_ids = [full.recommended_job.job_id, *[job.job_id for job in full.alternatives]]
    assert fast_ids[0] == full_ids[0]
    assert len(set(fast_ids) & set(full_ids)) >= 4
    assert engine.last_recommendation_stats["jobs_assessed"] == len(jobs)


def test_same_title_duplicates_collapse_but_different_roles_remain(tmp_path):
    description = "Proficient in Python and SQL is required for this data analyst opening."
    jobs = pd.DataFrame([
        {
            "job_id": "copy_a", "category": "INFORMATION-TECHNOLOGY", "job_title": "Data analyst",
            "job_description": description, "job_skill_set": "['python', 'sql']",
        },
        {
            "job_id": "copy_b", "category": "INFORMATION-TECHNOLOGY", "job_title": "Data analyst",
            "job_description": description, "job_skill_set": "['python', 'sql']",
        },
        {
            "job_id": "hr_copy", "category": "HR", "job_title": "Data analyst",
            "job_description": "Recruitment, onboarding and employee relations are required. The title is shared but the work is human resources.",
            "job_skill_set": "['recruitment', 'onboarding']",
        },
        {
            "job_id": "other_it", "category": "INFORMATION-TECHNOLOGY", "job_title": "Systems administrator",
            "job_description": "Maintain systems. Python is useful but the role is operations.",
            "job_skill_set": "['python']",
        },
    ])
    engine = _engine(jobs, cache_dir=tmp_path / "cache")
    result = engine.recommend("Skills: Python, SQL.", top_k=4, include_details=True)
    ids = [result.recommended_job.job_id, *[job.job_id for job in result.alternatives]]
    assert len(set(ids) & {"copy_a", "copy_b"}) == 1
    assert "hr_copy" in ids


def test_competency_cache_reused_and_invalidated(tmp_path):
    jobs = _make_jobs(per_category=2)
    cache = tmp_path / "cache"
    first = _engine(jobs, cache_dir=cache)
    files = list(cache.glob("job_competencies_*.json"))
    assert len(files) == 1
    stamp = files[0].stat().st_mtime_ns
    second = _engine(jobs, cache_dir=cache)
    assert files[0].stat().st_mtime_ns == stamp
    assert second._state.job_competencies[0][0].priority in {"required", "preferred", "not_required"}
    changed = jobs.copy()
    changed.loc[0, "job_description"] = "A completely different description that requires payroll only."
    _engine(changed, cache_dir=cache)
    assert len(list(cache.glob("job_competencies_*.json"))) == 2


def test_repeated_recommend_does_not_reencode_the_corpus(tmp_path):
    jobs = _make_jobs(per_category=3)
    engine = _engine(jobs, cache_dir=tmp_path / "cache")
    calls = []
    original = engine._state.embedder.encode

    def wrapped(texts):
        calls.append(len(list(texts)))
        return original(texts)

    engine._state.embedder.encode = wrapped
    engine.recommend(IT_STUDENT, top_k=3)
    engine.recommend(IT_STUDENT, top_k=3)
    assert calls
    assert max(calls) < len(jobs)
    assert sum(calls) < len(jobs)


def test_compare_jobs_accepts_every_supported_count(nlp):
    base = "Data analyst. Skills: python, sql, power bi."
    for count in range(2, 21):
        payload = [
            {"job_id": f"JD{index}", "job_title": f"Role {index}", "job_text": f"{base} Extra note {index}."}
            for index in range(count)
        ]
        response = nlp.compare_jobs(IT_STUDENT, payload)
        ranked = [response.recommended_job, *response.alternatives]
        assert len(ranked) == count
        assert [item.rank for item in ranked] == list(range(1, count + 1))
        scores = [item.match.match_score for item in ranked]
        assert scores == sorted(scores, reverse=True)
        assert {item.job_id for item in ranked} == {f"JD{index}" for index in range(count)}


def test_one_jd_match_and_incomplete_inputs(nlp):
    matched = nlp.match(IT_STUDENT, IT_JD, include_details=True)
    body = json.loads(matched.model_dump_json())
    assert {"match_score", "matched_skills", "missing_weak_skills", "recommended_learning", "competency_requirements"} <= set(body)
    assert 0 <= matched.match_score <= 100
    incomplete = nlp.match("Skills:", "We are hiring.")
    assert 0 <= incomplete.match_score <= 100
    noisy = nlp.recommend("hi!!! pythn maybe, teamwork???", top_k=2)
    assert noisy.recommended_job.job_id
    with pytest.raises(InputValidationError):
        nlp.recommend("   ")
    with pytest.raises(InputValidationError):
        nlp.match(IT_STUDENT, " ")
    with pytest.raises(InputValidationError):
        nlp.compare_jobs(IT_STUDENT, [{"job_text": "Only one"}])


def test_recommend_response_shape_and_academic_penalty_still_apply(tmp_path):
    jobs = _make_jobs(per_category=3)
    jobs.loc[0, "job_description"] = (
        "Data analyst requiring strong expertise in Physics. "
        "Responsibilities include python, sql, machine learning and data analysis."
    )
    jobs.loc[0, "job_title"] = "Physics-focused data analyst"
    engine = _engine(jobs, cache_dir=tmp_path / "cache")
    result = engine.recommend(
        "B.Tech Data Science. Skills: python, sql, machine learning, data analysis. Physics: 35%.",
        top_k=10,
        include_details=True,
    )
    found = next(job for job in [result.recommended_job, *result.alternatives] if job.job_id == str(jobs.iloc[0]["job_id"]))
    assert found.academic_penalty_percent == 40.0
    assert found.match_score == pytest.approx(max(0.0, found.base_match_score - 40.0))
    assert result.confidence.level in {"low", "medium", "high"}
    assert result.confidence.calibrated is False
    assert result.academic_records[0].subject == "Physics"
    payload = json.loads(result.model_dump_json(exclude_none=True))
    assert {"recommended_job", "alternatives", "confidence", "academic_records", "meta"} <= set(payload)
    assert "match_score" in payload["recommended_job"]
    assert parse_skills(jobs.iloc[0]["job_skill_set"])
