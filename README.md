# Group 8 — Placement & Career Intelligence NLP Asset

This repository retains the Week 2 baseline, Week 3 core pipeline and Week 4 evaluation, and includes the Week 6 reusable `career_nlp` asset as the current API-ready component. For the current asset's endpoints, schemas and implementation notes, see `WEEK6_ASSET_ENGINEERING.md`.

A simple browser UI for comparing multiple job descriptions is in `frontend/`. See `frontend/README.md` for startup instructions.

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

The job split uses `random_state=42` and stratifies by category. For the held-out evaluation, the Week 2 lexical baseline fits its unsupervised TF-IDF vocabulary and IDF weights on the training jobs only; no test-job text or relevance label is used to fit either model. The deployed search index may use the complete supplied corpus, because it is not a held-out evaluation.

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

The baseline ranks by **TF-IDF cosine similarity** only. Exact skill overlap is returned as an explanation, not mixed into the baseline score. It is therefore a simple lexical ranking baseline, not a trained supervised classifier.

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
- The Week 3 core Match Score: `0.63 × Skill Match % + 0.27 × Sentence-Transformer Similarity % + 0.10 × train-split Category Compatibility %`
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

Full documentation: `WEEK6_ASSET_ENGINEERING.md`.

## Current Week 6 functionality (asset version 1.4.5)

The reusable asset extracts academic records from student text and exposes them through `/v1/extract`, `/v1/cv/analyze`, `/v1/match`, `/v1/analyze`, `/v1/recommend`, and `/v1/compare-jobs`. It recognises subject percentages, fractions such as `45/100`, letter grades, GPA/CGPA out of 4 or 10, and GPA without a stated scale (assumed out of 4 for values up to 4, and out of 10 for values greater than 4). Numeric values are normalised to a 0–100 scale when valid; out-of-range values are not scored. The default poor-grade threshold is 50%, configurable with `CAREER_NLP_POOR_GRADE_THRESHOLD_PERCENT`. Overall GPA is reported but is not mapped to a specific weakness.

### General competency matching

The JD matcher now combines the existing job-skill vocabulary with generic language patterns to discover required competencies and knowledge areas from arbitrary descriptions. It does not depend on a fixed list of subjects or technologies, so the same mechanism can identify terms such as Chemistry, NLP, RAG, a domain-specific method, or an unfamiliar tool when the JD states them clearly. Rule-based language cues identify phrases such as `knowledge of Chemistry is nice to have`, `deep expertise in NLP is required`, and `strong knowledge of RAG is mandatory`; the existing embedding backend remains available as a fallback for semantic evidence.

Each match includes `competency_requirements`, with `kind` (`skill` or `knowledge_area`), `priority` (`required`, `preferred`, or `not_required`), `status` (`matched`, `missing`, or `not_evaluated`), and the source wording from the JD. Preferred requirements contribute less to Skill Match by default (`CAREER_NLP_PREFERRED_REQUIREMENT_WEIGHT=0.35`). Explicitly non-required requirements are excluded from Skill Match and learning gaps. For backward compatibility, matched knowledge areas appear in `matched_skills`, while missing required and preferred items appear in `missing_weak_skills` and `recommended_learning`; use `competency_requirements.kind` and `.priority` to interpret them precisely.

### Keep grades separate from competency evidence

A grade such as `NLP: 35%` is returned as an academic record, but that grade alone never proves the student knows NLP. If a required/core competency is missing, it is reported as missing and recommended for learning; when the same area has a poor recorded grade, the academic-risk adjustment is also applied. These are separate signals: one reflects JD competency coverage, the other reflects explicit academic evidence. A competency supported independently by the student's listed skills, coursework, experience, or project evidence can satisfy the requirement; a poor grade by itself is not treated as proof of a missing skill.

The academic base deduction is 40 points (`CAREER_NLP_ACADEMIC_SUBJECT_PENALTY`) and scales up to 60 when the weak subject/topic is estimated to be central to the role (`CAREER_NLP_ACADEMIC_CORE_DOMAIN_PENALTY`). The centrality assessment uses the **job description**, not its title, so a misleading title alone cannot make a subject core to the role. Optional language and explicit non-requirements suppress the academic adjustment. This remains an explainable heuristic rather than a validated prediction of performance; unusual wording and real applicant data should be used to evaluate it before operational use.

`POST /v1/compare-jobs` accepts one `student_text` plus 2–20 caller-supplied job descriptions, then returns a ranked recommendation and alternatives. The full input for a test covering required vs. preferred knowledge, generic topics, and a title/description mismatch is included at [`examples/sample_compare_jobs_domain_requirements.json`](examples/sample_compare_jobs_domain_requirements.json). Run it through `POST /v1/compare-jobs` in Swagger at `http://127.0.0.1:8000/docs`.

Run the reusable asset from the repository root with `pip install -e ".[all,dev]"`, then `python -m career_nlp serve --host 127.0.0.1 --port 8000`. The offline regression suite is `pytest -q test/test_career_nlp.py`.

## Local frontend

The lightweight CareerMatch frontend is in `frontend/`. It automatically picks the API flow based on the fields supplied:

- Student profile only: recommend jobs from the dataset.
- Student profile plus one job description: calculate that job's match score.
- Student profile plus multiple job descriptions: compare and rank the supplied jobs.

Run these in two terminals from the project root:

```bash
python -m career_nlp serve --host 127.0.0.1 --port 8000
```

```bash
python frontend/server.py
```

Then open `http://127.0.0.1:5500`. The frontend server proxies API calls to the backend, so the local browser setup does not need a CORS environment variable. See `frontend/README.md` for details.
