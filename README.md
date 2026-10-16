# Group 8 — Placement & Career Intelligence Baseline

This project implements the initial NLP baseline for the Group 8 Placement & Career Intelligence system.

## What the baseline does

The baseline supports two related workflows:

1. **Interactive matching:** a student profile/CV is matched against one job description and returns a match score, matched skills, missing skills, and learning recommendations.
2. **Initial evaluation:** every test student is scored against every test job, jobs are ranked, and Precision@5, Recall@5, and MRR are calculated using the student's expected job category as a proxy relevance label.

## Project structure

```text
SNLP-main/
├── dataset/
│   └── job_skill_set.parquet
├── base version/
│   ├── data/
│   ├── src/
│   │   ├── baseline.py                # Week 2 baseline (kept for comparison)
│   │   ├── inspect_data.py
│   │   ├── prepare_data.py
│   │   ├── run_baseline.py
│   │   ├── evaluation_baseline.py
│   │   └── week3/                     # Week 3 core NLP pipeline
│   │       ├── nlp_v1.py
│   │       ├── api.py
│   │       └── run_v1_demo.py
│   ├── test/
│   │   ├── test_students.csv
│   │   └── test_nlp_v1.py
│   └── requirements.txt
└── test.ipynb
```

## Setup

From the project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r "base version/requirements.txt"
```

## Prepare the data

Run once, or rerun whenever the source dataset/preparation code changes:

```bash
python "base version/src/inspect_data.py"
python "base version/src/prepare_data.py"
```

`prepare_data.py` creates:

- `base version/data/cleaned_jobs.csv`
- `base version/data/train_jobs.csv`
- `base version/data/test_jobs.csv`
- `base version/test/test_students.csv`

The original `job_id` is preserved so evaluation can identify individual jobs.

The job split uses `random_state=42` and stratifies by category. The TF-IDF vectorizer and skill vocabulary used by the evaluation are fitted from the training jobs only.

## Run the interactive baseline

```bash
python "base version/src/run_baseline.py"
```

Options:

```text
1. Type student profile
2. Use CV from base version/cv/
3. Run test_students.csv
```

PDF resumes use `pypdf`; DOCX resumes use `python-docx`.

## Run the initial evaluation

After preparing the data:

```bash
python "base version/src/evaluation_baseline.py"
```

The evaluation scores each test student against every test job and ranks the jobs by the same baseline scoring logic. It produces:

- `base version/test/initial_evaluation_results.csv`
- `base version/test/initial_evaluation_summary.csv`

Metrics:

- **Precision@5:** fraction of the top 5 jobs belonging to the student's expected category.
- **Recall@5:** fraction of all relevant test jobs retrieved in the top 5.
- **MRR:** reciprocal rank of the first relevant job.

The expected categories are proxy labels for the synthetic student profiles and should be described as such in the report.

## Baseline scoring

The baseline combines:

- 70% skill coverage
- 30% TF-IDF cosine similarity

The final score is therefore a heuristic ranking score, not a trained supervised classifier.

## Git hygiene

Do not commit Python cache files or virtual environments. Use the provided `.gitignore`.

---

# Week 3 — Core NLP Pipeline

The project now includes the Week 3 core NLP pipeline and FastAPI interface, in `base version/src/week3/`. The Week 2 baseline (`base version/src/baseline.py` and `evaluation_baseline.py`) stays outside that folder and remains available separately for comparison — it is **not** used to compute the Week 3 Match Score.

### Components
- Domain NER / information extraction (skills, degree + specialization, emails, phone numbers, experience mentions)
- A genuinely derived student profile representation (normalized education + skills + experience), used as the embedding input
- Job-category classification (auxiliary component)
- Sentence-Transformer semantic embeddings (the Week 3 embedding model), generated and stored for every job in the full cleaned dataset
- The Week 3 core Match Score: `0.70 × Skill Match % + 0.30 × Semantic Similarity %`
- Semantic ("RAG"-style) job retrieval over the full cleaned job dataset
- Extractive summarization (generic utility)
- PDF/DOCX CV processing

### Two API flows

- **JD provided** — `POST /nlp/match` with `student_text` + `job_text`. Returns only Match Score, Matched Skills, Missing/Weak Skills and Recommended Learning.
- **JD not provided** — `POST /nlp/recommend` with `student_text` only. Searches/ranks the full cleaned job dataset and returns the best-matching job (title, job_id), Match Score, Matched Skills, Missing/Weak Skills and Recommended Learning.

### API

Run:

```bash
pip install -r "base version/requirements.txt"
cd "base version/src/week3"
uvicorn api:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

The first run downloads the `all-MiniLM-L6-v2` Sentence-Transformer model, so an internet connection is needed the first time the Week 3 pipeline starts. Job embeddings for the full cleaned dataset are cached to `base version/data/week3_job_embeddings.npz` after the first run.

See `base version/V1_NLP_PIPELINE.md` for the component map, endpoints, limitations and final-API design.


---

# Week 6 — Asset Engineering (reusable NLP component)

The NLP model is now packaged as a reusable component, `career_nlp`
(`base version/src/career_nlp/`), with a typed Python API, a versioned REST API,
a CLI, structured input/output, confidence scores and full documentation.
Week 3 code is untouched.

```bash
pip install -e "base version[all,dev]"
python -m career_nlp recommend --student "B.Tech Data Science. python, sql, power bi"
career-nlp serve --port 8000        # Swagger at http://127.0.0.1:8000/docs
pytest "base version/test/test_career_nlp.py"
```

Full documentation: `base version/WEEK6_ASSET_ENGINEERING.md`.
