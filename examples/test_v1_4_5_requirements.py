"""Smoke-test the Week 6 academic/domain requirement behavior.

Start the API first:
    python -m career_nlp serve --host 127.0.0.1 --port 8000

Then run:
    python examples/test_v1_4_5_requirements.py
"""

import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_compare_jobs_domain_requirements.json"
URL = "http://127.0.0.1:8000/v1/compare-jobs"


def main() -> None:
    payload = json.loads(SAMPLE.read_text())
    request = Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        result = json.load(response)

    jobs = [result["recommended_job"], *result["alternatives"]]
    by_id = {job["job_id"]: job["match"] for job in jobs}

    core = by_id["TITLE_MISMATCH_CORE_NLP"]
    not_core = by_id["TITLE_MISMATCH_NOT_CORE"]
    data_ops = by_id["DATA_OPS"]

    core_requirements = {
        item["name"].casefold(): item for item in core["competency_requirements"]
    }
    not_core_requirements = {
        item["name"].casefold(): item for item in not_core["competency_requirements"]
    }

    assert core_requirements["nlp"]["priority"] == "required"
    assert core_requirements["nlp"]["status"] == "missing"
    assert core_requirements["rag"]["priority"] == "required"
    assert core_requirements["rag"]["status"] == "missing"

    assert core_requirements["chemistry"]["priority"] == "preferred"
    assert "chemistry" not in {x.casefold() for x in core["missing_weak_skills"]}
    assert "chemistry" in {x.casefold() for x in core["recommended_learning"]}

    assert core["academic_penalty_percent"] == 60.0
    assert not_core["academic_penalty_percent"] == 0.0
    assert "nlp" not in not_core_requirements or not_core_requirements["nlp"]["priority"] == "not_required"

    assert data_ops["academic_penalty_percent"] == 0.0
    assert "chemistry" not in {x.casefold() for x in data_ops["missing_weak_skills"]}

    print("PASS: required/preferred/not-required classification")
    print("PASS: title/description mismatch handling")
    print("PASS: poor-grade core-domain penalty")
    print("PASS: preferred knowledge excluded from required-gap output")
    print("PASS: recommended learning still includes preferred knowledge")
    print("Recommended job:", result["recommended_job"]["job_id"], "-", result["recommended_job"]["job_title"])


if __name__ == "__main__":
    main()
