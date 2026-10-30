"""Measure the pre-change recommendation path. Writes benchmarks/before.json."""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from career_nlp import CareerNLP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "before.json"

PROFILES = {
    "technical": (
        "B.Tech in Computer Science. Skills: Python, Java, SQL, Git, REST APIs. "
        "Projects: Built backend services in Python and wrote SQL queries for application data."
    ),
    "data": (
        "B.Tech in Data Science. Skills: Python, SQL. "
        "Projects: Wrote SQL queries and Python scripts for data reporting."
    ),
    "transferable": (
        "Business graduate. Skills: Excel, stakeholder communication, presentation, research, "
        "problem solving. Internship: analysed reports and prepared dashboards for managers."
    ),
    "missing_essentials": (
        "B.A. in English. Skills: communication, collaboration, writing, adaptability. "
        "No programming experience."
    ),
    "multi_category": (
        "Skills: Python, SQL, financial reporting, Excel, recruitment, CRM, machine learning, Power BI. "
        "Experience across analytics, finance operations and campus hiring support."
    ),
    "noisy": "hi!!! I want job. I know pythn and some sql??? worked on data stuff. also teamwork.",
}


def summarise(job):
    return {
        "job_id": job.job_id,
        "job_title": job.job_title,
        "category": job.category,
        "match_score": job.match_score,
        "base_match_score": job.base_match_score,
        "academic_penalty_percent": job.academic_penalty_percent,
        "matched_skills": job.matched_skills,
        "missing_weak_skills": job.missing_weak_skills,
        "recommended_learning": job.recommended_learning,
        "skill_match_percent": None if job.score_breakdown is None else job.score_breakdown.skill_match_percent,
        "semantic_similarity_percent": None if job.score_breakdown is None else job.score_breakdown.semantic_similarity_percent,
        "category_compatibility_percent": None if job.score_breakdown is None else job.score_breakdown.category_compatibility_percent,
        "required_skills_detected": None if job.score_breakdown is None else job.score_breakdown.required_skills_detected,
        "confidence_not_on_job": None,
    }


def main():
    print("IMPORT_DONE", flush=True)
    t0 = time.perf_counter()
    nlp = CareerNLP().warmup()
    warmup_s = time.perf_counter() - t0
    st = nlp._state
    print(f"WARMUP_DONE {warmup_s:.3f} jobs={len(st.job_ids)} backend={st.backend}", flush=True)

    idx = st.job_ids.index("job_00219")
    priorities = Counter(item.priority for item in st.job_competencies[idx])
    job_219 = [
        {"name": item.name, "priority": item.priority, "kind": item.kind, "evidence": item.evidence[:180]}
        for item in st.job_competencies[idx]
    ]

    assess_times = []
    original = CareerNLP._assess_competencies

    def wrapped(self, *args, **kwargs):
        started = time.perf_counter()
        result = original(self, *args, **kwargs)
        assess_times.append(time.perf_counter() - started)
        return result

    CareerNLP._assess_competencies = wrapped

    profile_results = {}
    first_latency = None
    for name, text in PROFILES.items():
        started = time.perf_counter()
        before = len(assess_times)
        result = nlp.recommend(text, top_k=5, include_details=True)
        elapsed = time.perf_counter() - started
        if first_latency is None:
            first_latency = elapsed
        jobs = [result.recommended_job, *result.alternatives]
        profile_results[name] = {
            "latency_s": round(elapsed, 4),
            "jobs_assessed": len(assess_times) - before,
            "processing_ms": result.meta.processing_ms,
            "confidence": result.confidence.model_dump(),
            "academic_records": [record.model_dump() for record in result.academic_records],
            "recommendations": [summarise(job) for job in jobs],
        }
        print(f"PROFILE {name} {elapsed:.3f}s top={jobs[0].job_id} {jobs[0].match_score}", flush=True)

    repeat_text = PROFILES["data"]
    repeat = []
    for i in range(7):
        started = time.perf_counter()
        nlp.recommend(repeat_text, top_k=5, include_details=True)
        elapsed = time.perf_counter() - started
        repeat.append(elapsed)
        print(f"REPEAT {i} {elapsed:.3f}s", flush=True)

    # Direct score for job_00219 using the same assessment the recommender stored.
    data_text = PROFILES["data"]
    profile = nlp._profile_from_fields(nlp._extract_fields(st, data_text))
    evidence = nlp._student_evidence_sentences(data_text, profile.academic_records)
    assessment = original(
        nlp,
        st,
        st.job_descriptions[idx],
        list(profile.skills),
        evidence,
        baseline_skills=sorted(st.job_skill_sets[idx]),
        competency_candidates=st.job_competencies[idx],
        semantic_fallback=False,
    )
    sem = float(nlp._cos_percent(nlp._embed(st, [f"{data_text} {profile.profile_summary}"]), st.index_vectors[idx:idx + 1])[0])
    category = nlp._category_probability_percent(st, data_text, st.job_categories[idx])
    base = round(nlp._combine(assessment.skill_match_percent, sem, category), 2)
    direct_219 = {
        "skill_match_percent": round(assessment.skill_match_percent, 2),
        "semantic_similarity_percent": round(sem, 2),
        "category_compatibility_percent": round(category, 2),
        "base_match_score": base,
        "matched_skills": assessment.matched,
        "missing_weak_skills": assessment.missing,
        "recommended_learning": nlp._order_learning(st, assessment.learning),
        "priorities": [
            {"name": item.name, "priority": item.priority, "status": item.status}
            for item in assessment.requirements
        ],
        "student_skills": profile.skills,
    }

    repeat_sorted = sorted(repeat)
    payload = {
        "conditions": {
            "backend": st.backend,
            "embedding_model": st.embedder.name,
            "jobs": len(st.job_ids),
            "top_k": 5,
            "include_details": True,
            "note": "Pre-change full-dataset competency assessment on every recommend() call.",
        },
        "warmup_s": round(warmup_s, 4),
        "first_request_s": round(first_latency, 4),
        "repeat_request_s": [round(value, 4) for value in repeat],
        "median_repeat_s": round(statistics.median(repeat), 4),
        "p95_repeat_s": round(repeat_sorted[max(0, int(len(repeat_sorted) * 0.95) - 1)], 4),
        "jobs_assessed_per_request": len(st.job_ids),
        "shortlist_size": len(st.job_ids),
        "assess_call_median_ms": round(statistics.median(assess_times) * 1000, 3) if assess_times else None,
        "assess_call_count": len(assess_times),
        "job_00219_extracted_priority_counts": dict(priorities),
        "job_00219_extracted": job_219,
        "job_00219_data_profile": direct_219,
        "profiles": profile_results,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
