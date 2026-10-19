# Week 4 — Baseline vs Improved Model Comparison

## Objective

Week 4 improves the Week 2 NLP matching baseline and compares both approaches
on the same held-out test set.

### Baseline
- TF-IDF text representation fitted on the training-job split
- Final score = `TF-IDF cosine similarity`

### Improved model
- Sentence-Transformer (`all-MiniLM-L6-v2`) semantic representation
- Exact skill-coverage signal
- Job-category compatibility from a classifier trained only on the train split
- Final score = `63% skill coverage + 27% Sentence-Transformer similarity + 10% category compatibility`

This compares a simple lexical baseline with a hybrid core model. The core
retains dense Sentence-Transformer matching and adds small, independently
trained category-compatibility evidence without using the test students'
labels.

## Evaluation

Both models are evaluated using:
- 16 synthetic student profiles
- 234 held-out test jobs
- The same expected-category relevance labels
- Precision@5
- Recall@5
- Mean Reciprocal Rank (MRR)

For a leakage-free comparison, the baseline TF-IDF vocabulary and IDF weights,
and the core category classifier, are fitted only on the training-job split.
The held-out jobs are used only for ranking and metric calculation. The
Sentence-Transformer is a fixed pretrained encoder; it is not fitted on either
split or on the proxy relevance labels.

The expected job category is a **proxy relevance label** for these synthetic
profiles; it is not a human-rated ground truth.

## Latest measured result

The current generated comparison (`test/week4_model_comparison.csv`) reports:

| Metric | TF-IDF baseline | Core hybrid | Change |
|---|---:|---:|---:|
| Precision@5 | 0.8625 | 0.8750 | +0.0125 |
| Recall@5 | 0.0925 | 0.0938 | +0.0014 |
| MRR | 0.8906 | 0.9688 | +0.0781 |

Recall@5 is necessarily small because five results are retrieved from
categories containing roughly 44–48 relevant jobs. Treat these as proxy-label
benchmark results, not a claim of production accuracy.

## Run

From the project root:

```powershell
pip install -r "base version/requirements.txt"
python "base version/src/week4/evaluation_week4.py"
```

or:

```powershell
python "base version/src/week4/run_week4_comparison.py"
```

The first improved-model run downloads `all-MiniLM-L6-v2`. The model is cached
inside:

```text
base version/data/sentence_transformer_model/
```

Job embeddings are reused from:

```text
base version/data/week3_job_embeddings.npz
```

## Output files

The comparison creates:

- `base version/test/week4_comparison_results.csv`
- `base version/test/week4_comparison_summary.csv`
- `base version/test/week4_model_comparison.csv`

The last file contains the baseline value, improved-model value, absolute
change and percentage change for Precision@5, Recall@5 and MRR.

## What to show in the submission

Use the generated comparison table as the main deliverable. Then explain that
the improved approach uses contextual sentence embeddings, so semantically
similar wording can be matched even when the exact words do not overlap as
strongly as they must with TF-IDF.

Do not claim that the improved model is better until the evaluation script has
been run and the measured metrics are available.

## Running Week 4 in VS Code

From the project root, run:

```powershell
python "base version\src\week4\run_week4_comparison.py"
```

The launcher checks for `sentence-transformers` first and installs the packages
from `base version\requirements.txt` into the active Python environment if the
dependency is missing. The first successful run also downloads
`all-MiniLM-L6-v2`; subsequent runs reuse the local model cache at
`base version\data\sentence_transformer_model\`.

If installation fails, make sure VS Code is using the intended Python interpreter
and that the machine has internet access. You can also install manually with:

```powershell
python -m pip install -r "base version\requirements.txt"
```

Then rerun the Week 4 command.
