# Week 4 — Baseline vs Improved Model Comparison

## Objective

Week 4 improves the Week 2 NLP matching baseline and compares both approaches
on the same held-out test set.

### Baseline
- TF-IDF text representation
- Skill coverage
- Final score = `70% skill coverage + 30% TF-IDF cosine similarity`

### Improved model
- Sentence-Transformer (`all-MiniLM-L6-v2`) semantic representation
- The same skill-coverage signal
- Final score = `70% skill coverage + 30% Sentence-Transformer semantic similarity`

This isolates the main improvement requested for Week 4: replacing sparse
TF-IDF semantic matching with dense Sentence-Transformer semantic matching.

## Evaluation

Both models are evaluated using:
- 16 synthetic student profiles
- 234 held-out test jobs
- The same expected-category relevance labels
- Precision@5
- Recall@5
- Mean Reciprocal Rank (MRR)

The expected job category is a **proxy relevance label** for these synthetic
profiles; it is not a human-rated ground truth.

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