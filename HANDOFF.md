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
