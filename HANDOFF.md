# Handoff: Group 8 Placement & Career Intelligence

## Current outcome

The requested Week 4 ordering is now verified with the real local
`all-MiniLM-L6-v2` Sentence-Transformer model on 16 synthetic student profiles
and 234 held-out jobs:

| Metric | TF-IDF baseline | Core hybrid | Change |
|---|---:|---:|---:|
| Precision@5 | 0.8625 | 0.8750 | +0.0125 |
| Recall@5 | 0.0925 | 0.0938 | +0.0014 |
| MRR | 0.8906 | 0.9688 | +0.0781 |

The generated source of truth is `test/week4_model_comparison.csv`; per-student
and top-five outputs are in `test/week4_comparison_summary.csv` and
`test/week4_comparison_results.csv`.

## Important evaluation rule

`src/week4/evaluation_week4.py` now fits the baseline TF-IDF vocabulary and
IDF weights on `data/train_jobs.csv` only. The test jobs are used solely for
ranking and metrics. This avoids leaking held-out text statistics into the
baseline. The core's category classifier is likewise training-split only; the
Sentence-Transformer is fixed pretrained model, not fit on evaluation labels.

The deployed `career_nlp` asset is separate from the evaluation: it indexes the
complete supplied job corpus, which is appropriate for deployment. Its score is
the documented hybrid of skill coverage (0.63), Sentence-Transformer semantic
similarity (0.27), and classifier category compatibility (0.10).

## What was verified

* `python src/week4/evaluation_week4.py` completed with the real local model
  and regenerated all three Week 4 CSVs.
* `python src/week2/evaluation_baseline.py` regenerated the consistent,
  training-split-only Week 2 baseline outputs (0.8625 P@5, 0.0925 R@5,
  0.8906 MRR).
* The local environment has `sentence-transformers 6.1.0` and CPU-only
  `torch 2.14.1`.
* The 51-test reusable-asset/API suite passed outside the restrictive temporary
  directory sandbox: `51 passed in 3.08s`.
* A sandboxed pytest run initially showed temporary-directory permission errors,
  not product-code failures. If that recurs, use a workspace-local base temp
  directory:

  ```powershell
  $env:PYTHONPATH = 'src'
  python -m pytest -q --basetemp .pytest-tmp-run
  ```

## Files changed in this continuation

* `src/week2/evaluation_baseline.py` and `src/week4/evaluation_week4.py` —
  leakage-free baseline fitting.
* `WEEK4_MODEL_COMPARISON.md` — method and current metrics.
* `README.md` — describes the strict evaluation split accurately.
* This handoff file.

## Useful next commands

```powershell
python src/week4/evaluation_week4.py
$env:PYTHONPATH = 'src'
python -m pytest -q --basetemp .pytest-tmp-run
python examples/parity_check_week3.py
python examples/validate_confidence.py --backend sentence-transformers
```

The repository is not a Git working tree, so use normal file inspection rather
than Git diff/status for change review.


## Week 6 competency extraction update (asset 1.4.5)

* Added structured competency results (`competency_requirements`) that separate skills from broader knowledge areas and classify each as required, preferred or not required, with matched/missing/not-evaluated status and JD evidence.
* Generic cue patterns detect unfamiliar topics/skills from JD wording; there is no topic-specific list. Preferred requirements count at a configurable lower weight (`CAREER_NLP_PREFERRED_REQUIREMENT_WEIGHT`, default `0.35`). Optional items are surfaced as preferred rather than mandatory, and explicit non-requirements are excluded from match gaps.
* Poor grades are not treated as skill evidence. Missing required subject/topic knowledge appears in `missing_weak_skills` and `recommended_learning`, while a poor grade in the same strongly required/core domain can also trigger a separate academic-risk adjustment. Demonstrated evidence in skills/projects/experience can satisfy a competency and suppress that academic adjustment for the same demonstrated area.
* JD title/body mismatch is handled by assessing academic core-domain relevance from the description only. The title can still contribute to overall semantic matching, but cannot alone trigger core-domain penalties.
* Sample request: `examples/sample_compare_jobs_domain_requirements.json` (covers title mismatch in both directions, required NLP/RAG, preferred Chemistry, and irrelevant Chemistry). Run via `POST /v1/compare-jobs` at `/docs`.
* Verified: `python -m pytest -q test/test_career_nlp.py` — **103 passed** with the offline LSA backend. This is regression testing, not a labelled real-world accuracy benchmark.

## Historical Week 6 core-domain penalty update (asset 1.4.3; superseded by 1.4.4)

* Introduced a 40-point base academic mismatch adjustment and up to 60 points for a poor grade in a required/core-domain subject.
* Asset 1.4.4 removes the earlier title-based academic domain signal so conflicting titles cannot override the job description.


## Week 6 academic-grade matching update (asset 1.4.0)

* `career_nlp` now extracts explicit subject-specific grades/scores from CV text and returns `academic_records` in extraction, CV-analysis, matching, and recommendation outputs.
* Poor results are flagged by explicit poor/fail wording, D/E/F/fail labels, or scores at/below 50%.
* A 35-point penalty was initially applied when the JD signalled a strong/mandatory requirement for the same subject; subject mentions alone and explicit non-requirements did not trigger it.
* `POST /v1/compare-jobs` accepts one student profile and 2–20 caller-supplied job descriptions, then ranks them using the existing hybrid match score with academic adjustments.
* Asset 1.4.0 regression suite: 86 tests passed locally with the offline LSA backend, including grade-format and JD-requirement phrasing tests.


## Week 6 academic-grade NLP update (asset 1.4.0)

- Grade extraction supports subject percentages, letter grades, subject grades, GPA/CGPA out of 4 or 10, and unscaled GPA (assumed out of 4).
- Grade threshold and scoring penalty are configurable with `CAREER_NLP_POOR_GRADE_THRESHOLD_PERCENT` and `CAREER_NLP_ACADEMIC_SUBJECT_PENALTY`.
- Academic mismatch analysis uses explicit requirement/negation rules first, then a cached embedding-based comparison between JD clauses and positive/neutral requirement examples.
- Overall GPA is returned as an academic record but cannot trigger a subject-specific penalty.
- Existing routes continue to use the shared engine implementation; `POST /v1/compare-jobs` ranks 2–20 supplied JDs.


## Week 6 GPA scale inference update (asset 1.4.1)

- For GPA/CGPA with no explicit denominator, values up to and including 4 default to a 4-point scale.
- Values greater than 4 and up to 10 default to a 10-point scale. An explicitly stated scale always takes precedence; out-of-range values remain visible but are not normalised.
- Regression tests cover GPA values 3.7, 4.0, 4.1, 7.8, 8.6, 10, and an invalid 11.2.

- Version 1.4.1 follow-up tests also verify subject-specific unscaled GPA inference and confirm that an explicitly supplied denominator always takes precedence.


## Week 6 academic-heading and ranking fix (asset 1.4.2)

* Fixed parsing when CV grades follow a heading such as `Academic record:`; the first subject grade is now retained.
* Raised the default configurable academic mismatch adjustment to 40 points so the demonstrated mandatory-Physics JD is down-ranked when the student has a 42% Physics grade. A JD explicitly saying Physics expertise is not required receives no Physics penalty.
* Verified the exact sample profile and three JDs against the bundled 1,167-job corpus using offline LSA; the Business Intelligence Analyst ranked first.
* All 97 Week 6 asset tests pass. The older Week 3 test suite requires `sentence-transformers`, which is unavailable in this execution environment.

## Current Week 6 requirement classification and scoring cleanup (asset 1.4.5)

* Preferred competencies are not included in `missing_weak_skills`; they remain available in `competency_requirements` and `recommended_learning`.
* `required_skills_detected` now counts required competencies only.
* Job-role/domain labels are not automatically treated as candidate competencies unless the description provides explicit requirement evidence.
* Generic competency extraction remains enabled for user-supplied JDs while bundled jobs continue using curated skill metadata for efficient warmup.
