"""Measure the two-stage recommender. Writes benchmarks/after.json.

Uses the same profiles and backend as benchmarks/before_benchmark.py.
Full-corpus ranking is timed separately and is not included in the latency stats.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from before_benchmark import PROFILES, summarise  # noqa: E402
from career_nlp import CareerNLP  # noqa: E402

OUT = ROOT / "benchmarks" / "after.json"
BEFORE = ROOT / "benchmarks" / "before.json"


def overlap(left, right):
    return len(set(left) & set(right))


def main():
    print("IMPORT_DONE", flush=True)
    t0 = time.perf_counter()
    nlp = CareerNLP().warmup()
    warmup_s = time.perf_counter() - t0
    st = nlp._state
    print(f"WARMUP_DONE {warmup_s:.3f} jobs={len(st.job_ids)} backend={st.backend}", flush=True)

    profile_results = {}
    first_latency = None
    for name, text in PROFILES.items():
        started = time.perf_counter()
        result = nlp.recommend(text, top_k=5, include_details=True)
        elapsed = time.perf_counter() - started
        if first_latency is None:
            first_latency = elapsed
        jobs = [result.recommended_job, *result.alternatives]
        stats = dict(nlp.last_recommendation_stats)
        stats.pop("assessed_job_ids", None)
        profile_results[name] = {
            "latency_s": round(elapsed, 4),
            "jobs_assessed": nlp.last_recommendation_stats["jobs_assessed"],
            "shortlist_size": nlp.last_recommendation_stats["shortlist_size"],
            "processing_ms": result.meta.processing_ms,
            "confidence": result.confidence.model_dump(),
            "recommendations": [summarise(job) for job in jobs],
        }
        print(
            f"PROFILE {name} {elapsed:.3f}s assessed={profile_results[name]['jobs_assessed']} "
            f"top={jobs[0].job_id} {jobs[0].match_score}",
            flush=True,
        )

    repeat = []
    for i in range(7):
        started = time.perf_counter()
        nlp.recommend(PROFILES["data"], top_k=5, include_details=True)
        elapsed = time.perf_counter() - started
        repeat.append(elapsed)
        print(f"REPEAT {i} {elapsed:.3f}s", flush=True)

    quality = {}
    for name, text in PROFILES.items():
        fast = nlp.recommend(text, top_k=5, include_details=True)
        full = nlp.recommend(text, top_k=5, include_details=True, full_ranking=True)
        fast_ids = [fast.recommended_job.job_id, *[job.job_id for job in fast.alternatives]]
        full_ids = [full.recommended_job.job_id, *[job.job_id for job in full.alternatives]]
        quality[name] = {
            "fast_ids": fast_ids,
            "full_ids": full_ids,
            "top1_agree": fast_ids[0] == full_ids[0],
            "top5_overlap": overlap(fast_ids, full_ids),
            "fast_scores": [fast.recommended_job.match_score, *[job.match_score for job in fast.alternatives]],
            "full_scores": [full.recommended_job.match_score, *[job.match_score for job in full.alternatives]],
        }
        print(f"QUALITY {name} top1={quality[name]['top1_agree']} overlap={quality[name]['top5_overlap']}", flush=True)

    idx = st.job_ids.index("job_00219")
    data_text = PROFILES["data"]
    profile = nlp._profile_from_fields(nlp._extract_fields(st, data_text))
    evidence = nlp._student_evidence_sentences(data_text, profile.academic_records)
    assessment = nlp._assess_competencies(
        st,
        st.job_descriptions[idx],
        list(profile.skills),
        evidence,
        baseline_skills=sorted(st.job_skill_sets[idx]),
        competency_candidates=st.job_competencies[idx],
        semantic_fallback=False,
    )
    semantic_profile = f"{data_text} {profile.profile_summary}"
    sem = float(nlp._cos_percent(nlp._embed(st, [semantic_profile]), st.index_vectors[idx:idx + 1])[0])
    category = nlp._category_probability_percent(st, data_text, st.job_categories[idx])
    base = round(nlp._combine(assessment.skill_match_percent, sem, category), 2)
    direct_219 = {
        "skill_match_percent": round(assessment.skill_match_percent, 2),
        "semantic_similarity_percent": round(sem, 2),
        "category_compatibility_percent": round(category, 2),
        "base_match_score": base,
        "matched_skills": assessment.matched,
        "missing_weak_skills": assessment.missing,
        "recommended_learning": assessment.learning,
        "priorities": [
            {"name": item.name, "priority": item.priority, "status": item.status}
            for item in assessment.requirements
        ],
        "student_skills": profile.skills,
        "priority_counts": dict(Counter(item.priority for item in st.job_competencies[idx])),
    }
    print("JOB219", base, direct_219["skill_match_percent"], direct_219["missing_weak_skills"], flush=True)

    before_profiles = {}
    if BEFORE.exists():
        before = json.loads(BEFORE.read_text(encoding="utf-8"))
        for name, payload in before["profiles"].items():
            before_ids = [item["job_id"] for item in payload["recommendations"]]
            after_ids = [item["job_id"] for item in profile_results[name]["recommendations"]]
            before_profiles[name] = {
                "before_ids": before_ids,
                "after_ids": after_ids,
                "top1_same_as_old_ranker": before_ids[0] == after_ids[0],
                "top5_overlap_with_old_ranker": overlap(before_ids, after_ids),
            }

    warm = subprocess.run(
        [sys.executable, "-c", "import time\nfrom career_nlp import CareerNLP\n"
         "t=time.perf_counter()\nCareerNLP().warmup()\nprint(f'{time.perf_counter()-t:.4f}')"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    warm_s = None
    if warm.returncode == 0:
        for line in reversed(warm.stdout.splitlines()):
            try:
                warm_s = float(line.strip())
                break
            except ValueError:
                continue
    else:
        print(warm.stderr[-500:], flush=True)

    repeat_sorted = sorted(repeat)
    payload = {
        "conditions": {
            "backend": st.backend,
            "embedding_model": st.embedder.name,
            "jobs": len(st.job_ids),
            "top_k": 5,
            "include_details": True,
            "candidate_shortlist_size": nlp.config.candidate_shortlist_size,
            "retrieval_skill_slots": nlp.config.retrieval_skill_slots,
            "recommendation_assess_pool": nlp.config.recommendation_assess_pool,
            "displayed_skill_gaps": nlp.config.displayed_skill_gaps,
            "note": "Post-change two-stage recommend() on the final code. Embedding and competency caches were already on disk, so warmup_s is a process cold start with those caches. warm_startup_s is a second process under the same conditions. Full-corpus ranking is excluded from the latency stats.",
        },
        "warmup_s": round(warmup_s, 4),
        "warm_startup_s": None if warm_s is None else round(warm_s, 4),
        "first_request_s": round(first_latency, 4),
        "repeat_request_s": [round(value, 4) for value in repeat],
        "median_repeat_s": round(statistics.median(repeat), 4),
        "p95_repeat_s": round(repeat_sorted[max(0, int(len(repeat_sorted) * 0.95) - 1)], 4),
        "jobs_assessed_per_request": profile_results["data"]["jobs_assessed"],
        "shortlist_size": profile_results["data"]["shortlist_size"],
        "job_00219_data_profile": direct_219,
        "profiles": profile_results,
        "full_ranking_agreement": quality,
        "comparison_with_previous_ranker": before_profiles,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
