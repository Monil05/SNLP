# Week 6 — Asset Engineering: `career_nlp`

**Deliverable:** the Week 3–5 NLP model converted into a reusable component with
(1) a Python module + REST API, (2) structured, validated input/output,
(3) documentation, and (4) confidence scores.

Package: `base version/src/career_nlp/` · Asset version `1.1.0` · Schema version `1.0`

---

## 1. What it is

One class, `CareerNLP`, that owns the job dataset, skill vocabulary, category
classifier, embedding backend and job index, and exposes:

| Method | REST route | Purpose | Confidence |
|---|---|---|---|
| `match(student, jd)` | `POST /v1/match` | **JD provided** – Match Score, matched / missing skills, learning | evidence heuristic |
| `recommend(student)` | `POST /v1/recommend` | **JD not provided** – best job + alternatives from the full dataset | evidence heuristic + category agreement |
| `analyze(student, jd=None)` | `POST /v1/analyze` | Picks the flow automatically | per flow |
| `extract(text)` | `POST /v1/extract` | Skills, degrees (+specialization), emails, phones, experience | per entity |
| `classify(text)` | `POST /v1/classify` | Job category | classifier probability |
| `retrieve(query)` | `POST /v1/retrieve` | Semantic job retrieval (RAG retrieval stage) | similarity score |
| `similarity(a, b)`, `embed(texts)`, `summarize(text)` | `/v1/similarity`, `/v1/embed`, `/v1/summarize` | Utilities | n/a (see §5) |
| `analyze_cv(path)` / `analyze_cv_bytes` | `POST /v1/cv/analyze` | PDF/DOCX → text, entities, profile | per entity |
| `info()` | `GET /v1/info`, `GET /v1/health` | Service/model metadata | – |

**Match Score** is the Week 3/4 hybrid: `0.63 × Skill Match % + 0.27 × Semantic Similarity % + 0.10 × Category Compatibility %` (all 0–100; weights configurable). Category compatibility is the profile classifier's probability for a job's category. Week 4 fits that classifier on the training split for fair evaluation; the reusable asset fits it on the complete supplied job corpus for deployment.

## 2. Install & run

```bash
pip install -e "base version[all,dev]"        # or: pip install -r "base version/requirements.txt"
```

Extras: `embeddings` (sentence-transformers), `api` (FastAPI/uvicorn), `cv` (pypdf/python-docx), `dev` (pytest/httpx).

**Python**
```python
from career_nlp import CareerNLP, AssetConfig

nlp = CareerNLP()                      # lazy; first call (or .warmup()) builds the index
m = nlp.match(student_text, job_text)  # -> MatchResponse (Pydantic)
print(m.match_score, m.confidence.level, m.recommended_learning)

r = nlp.recommend(student_text, top_k=3)
print(r.recommended_job.job_title, r.confidence.level, [a.job_title for a in r.alternatives])
```

**CLI** (`python -m career_nlp …` or `career-nlp …`; JSON on stdout, errors as JSON on stderr, exit code 2)
```bash
python -m career_nlp match --student-file cv.txt --job-file jd.txt --details
python -m career_nlp recommend --student "B.Tech Data Science. python, sql" --top-k 3
python -m career_nlp cv path/to/cv.pdf
python -m career_nlp serve --port 8000
```

**REST** – `career-nlp serve` (or `uvicorn career_nlp.api:app` from `base version/src`); Swagger at `/docs`; contract in `docs/openapi.json`.
```bash
curl -s -X POST localhost:8000/v1/recommend -H 'content-type: application/json' \
     -d '{"student_text":"MBA Finance. budgeting, forecasting, excel","top_k":2}'
```

**Reusing with your own data** – no source edits needed:
```python
nlp = CareerNLP(AssetConfig(jobs_path="my_jobs.csv", embedding_backend="auto"))   # or pass jobs=DataFrame
```
Required columns: `job_id, category, job_title, job_description, job_skill_set` (list-string or `;`-separated).
Custom embedder: pass any object with `name`, `dimensions`, `encode(texts) -> (n,d) L2-normalised float32` as `embedder=`.

## 3. Configuration

`AssetConfig` (or env vars via `AssetConfig.from_env()`):

| Field | Env var | Default |
|---|---|---|
| `jobs_path` | `CAREER_NLP_JOBS_PATH` | `base version/data/cleaned_jobs.csv` |
| `cache_dir` (job-embedding index) | `CAREER_NLP_CACHE_DIR` | `base version/data/career_nlp_cache` |
| `model_cache_dir` | `CAREER_NLP_MODEL_CACHE_DIR` | `base version/data/sentence_transformer_model` |
| `embedding_backend` | `CAREER_NLP_BACKEND` | `sentence-transformers` (`lsa`, `auto` also allowed) |
| `embedding_model` | `CAREER_NLP_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `skill_weight` / `semantic_weight` / `category_weight` | – | `0.63` / `0.27` / `0.10` (must sum to 1) |
| `max_text_chars`, `max_cv_bytes`, `max_top_k` | – | 50 000 / 5 MB / 20 |
| `warmup_on_start` (API) | `CAREER_NLP_WARMUP` | `true` |

API-only: `CAREER_NLP_API_KEY` (if set, all `/v1` routes except `/v1/health` need header `X-API-Key`), `CAREER_NLP_CORS_ORIGINS` (comma-separated; default none).

**Backends.** `sentence-transformers` is the Week 3 core model. If a local Hugging Face snapshot already exists in `model_cache_dir` it is loaded from disk (no network). `lsa` (TF-IDF + SVD) is an **offline/CI fallback only** — its scores are not comparable with Week 3. `auto` uses sentence-transformers if available, else LSA and puts a warning in `meta.warnings`.

## 4. Structured input / output

All requests and responses are Pydantic v2 models (`career_nlp/schemas.py`; `extra="forbid"` so typos in request fields fail loudly). JSON Schemas: `docs/schemas/`. Real sample outputs: `docs/sample_outputs/`.

Conventions: `match_score` and `*_percent` are 0–100; probabilities and confidence are 0–1; skill collections are **lists** (Week 3 used `"a; b; c"` strings — `CareerNLP.to_week3_dict()` converts back).

### `POST /v1/match`
Request: `{"student_text": str, "job_text": str, "include_details": false}`

```json
{
  "match_score": 70.06,
  "matched_skills": ["communication", "power bi", "python", "sql"],
  "missing_weak_skills": ["communication skills", "tableau"],
  "recommended_learning": ["communication skills", "tableau"],
  "confidence": {"score": 1.0, "level": "high", "method": "evidence_heuristic_v1", "calibrated": false},
  "meta": {"asset_version": "1.1.0", "schema_version": "1.0",
           "model": {"embedding_backend": "lsa", "embedding_model": "lsa-256", "jobs_indexed": 1167},
           "processing_ms": 88.4, "warnings": []}
}
```
`include_details=true` adds `score_breakdown` (skill %, semantic %, weights, #required skills), `student_profile` (the derived profile that is embedded) and `confidence.factors`. By default these internals stay hidden, preserving the Week 3 contract that the JD-provided flow exposes only score, matched, missing and learning (+ the new confidence/meta).

### `POST /v1/recommend`
Request: `{"student_text": str, "top_k": 3, "include_details": false}` → `recommended_job` (`rank, job_id, job_title, category, match_score, matched_skills, missing_weak_skills, recommended_learning`), `alternatives[]` (same shape), `confidence`, `meta`.
Job titles are cleaned (levels, requisition codes, parentheticals removed); alternatives skip titles already shown. `job_id`/`category` are returned (Week 3's docs promised `job_id` but its code dropped it).

### Errors
Every failure uses one envelope and a stable `code`:
```json
{"error": {"code": "invalid_input", "message": "'student_text' must not be empty.", "details": {"field": "student_text"}, "request_id": "bef3328f8ff8"}}
```
| HTTP | code | When |
|---|---|---|
| 401 | `unauthorized` | API key set but missing/wrong |
| 422 | `invalid_request` | body failed schema validation |
| 422 | `invalid_input` | blank/oversized text, bad `top_k` |
| 422 | `cv_processing_failed` | unsupported/corrupt/empty/oversized CV |
| 503 | `model_load_failed` | dataset or embedding model unavailable |

Responses carry `X-Request-ID` and `X-Process-Time-ms`. Python raises the same exceptions (`InputValidationError`, `CVProcessingError`, `ModelLoadError`, all subclasses of `CareerNLPError`).

## 5. Confidence scores

`Confidence = {score 0–1, level low|medium|high, method, calibrated, factors?}`. **Read `calibrated` and `method`: only the classifier score is a probability, and none is a guarantee.**

| Where | Meaning | How it is computed | Levels |
|---|---|---|---|
| `classify` | Classifier's predicted probability | Logistic-regression `predict_proba` (top class) | high ≥ 0.60, medium ≥ 0.40, else low |
| `match` | **Evidence sufficiency** — how much signal the score rests on | `sqrt(jd_evidence × profile_evidence)`, each `min(1, #recognised skills / 5)` | high ≥ 0.70, medium ≥ 0.40 |
| `recommend` | Evidence sufficiency + category consensus | `0.30·category_agreement_top5 + 0.25·top-job skill coverage + 0.25·profile evidence + 0.20·score margin(÷15)`; capped to *low* if < 50 % of the top-5 share the top job's category | as above |
| `extract` | Rule-based per-entity confidence | skills: 0.90 base, +0.05 repeated, +0.05 multi-word, −0.05 3-char, −0.25 ≤2-char (`r`, `c`); degrees 0.75–0.95; emails 0.98; phones 0.65–0.90 | numeric only |
| `retrieve`, `similarity`, `summarize` | Not a confidence; the cosine/centroid score is returned as `similarity` / `relevance` | | |

**What was validated, and what was not**

* *Classifier thresholds* come from a held-out check (25 % stratified split of the 1 167 jobs, 292 test jobs, title + description text): overall accuracy 96.9 %; accuracy was 100 % at p ≥ 0.60 (n = 132), 96–98 % for 0.40–0.60 (n = 121) and ~87 % below 0.40 (n = 39). Caveat: those are job-length texts; short student profiles will tend to get flatter probabilities. The probability itself is not recalibrated (`calibrated: false`).
* *Recommend confidence* was checked on the project's 16 synthetic students (`examples/validate_confidence.py --backend lsa`): top-1 category accuracy 13/16 (0.81). By level: **low 0/2 correct, medium 12/13, high 1/1**; mean confidence 0.64 for correct vs 0.44 for wrong top-1. **Treat this as weak, in-sample evidence.** Before the category-split cap the signal was weak (point-biserial r ≈ 0.33; the two "high" cases split 1 right / 1 wrong); the cap was added *after* seeing that all misses had a split top-5, so the 16-student result flatters it. n = 16, proxy labels, LSA backend.
* *Not validated:* `match` confidence has no ground truth to check against; thresholds for `high/medium` are judgement calls. **Re-run `examples/validate_confidence.py` with the Sentence-Transformer backend and, ideally, a human-labelled set before relying on the levels.**

## 6. Architecture

```
            ┌──────────── REST (api.py) ─ CLI (cli.py) ─ Python import ────────────┐
            │                      typed requests / responses (schemas.py)          │
            └───────────────────────────────┬──────────────────────────────────────┘
                                      CareerNLP (engine.py)
   text.py  extraction.py  confidence.py  embedders.py ── ST | LSA | custom
   (clean, skill vocab,  (degrees, contacts, (rules,       job index cached in cache_dir,
    SkillExtractor)       experience, skills) thresholds)   keyed by hash(model + job texts)
```
* **Lazy, thread-safe init** (double-checked lock); stateless afterwards, so a single instance can serve concurrent requests (sync routes run in FastAPI's thread pool).
* **No sys.path hacks / hard-coded paths** (Week 3 needed both); paths come from `AssetConfig`.
* **Self-contained**: does not import Week 2/3 code. Skill-matching speed-up: substring pre-filter before the word-boundary regex (same results, far fewer regex runs on the ~5k-skill vocabulary).
* Typical latency on the full dataset, LSA backend: ~45 ms `recommend`, ~90 ms `match` (single request, CPU). Warm-up ≈ 16 s here. Sentence-Transformer timings were not measured (see §9).

## 7. Changes vs Week 3 (deliberate; scores can differ)

1. **Degree extraction bug fixed.** Week 3 matched `BE/ME/MS` case-insensitively, so *"to be a data analyst, contact me"* produced degrees `B.E.` and `M.E. in anytime`, and *"MS Office"* produced `M.S. in Office` — and those strings flowed into the embedded profile text. Ambiguous abbreviations now need dots or an explicit `in <subject>` / `(<subject>)`. CFA/CPA no longer take a "specialization".
2. **Phone numbers:** runs of years (`2019 2020 2021`) are no longer phones; 10–13 digits required.
3. **Classifier no longer trains on text containing its own label** (Week 3 included the category name in the training text); it uses title + description.
4. **`recommended_learning` is ordered** by how many jobs in the dataset require the skill (most in-demand first); `missing_weak_skills` stays alphabetical. Same set of skills as Week 3.
5. **Alternatives** in `recommend` skip titles already shown.
6. **DOCX CVs:** table cells are read too (many CVs keep skills in tables).
7. Summariser's TF-IDF is fit on all jobs (Week 3 used the train split).
8. Job-embedding cache key now hashes model + job text (Week 3 keyed only on ids), so edited descriptions invalidate it.

Legacy paths are unaffected: `src/week3/` and its API still run as before; `CareerNLP.to_week3_dict()` reproduces the old response shape.

## 8. Testing

`pytest "base version/test/test_career_nlp.py"` — 51 tests, ~3 s, fully offline (synthetic job set, LSA or a stub embedder): output structure and score formula, confidence rules, degree/phone regression cases, validation and error envelopes, config/env, embedder plug-in + disk cache + cache invalidation, Sentence-Transformer loader (local-snapshot vs download branch, via a stub), CV (DOCX incl. tables, bad inputs), every REST route, API-key auth, health before/after warm-up, CLI.
Also run: `examples/quickstart.py`, `examples/validate_confidence.py`, `examples/parity_check_week3.py`, `examples/generate_docs.py` (regenerates `docs/`).

## 9. Known limitations / not yet verified

* **The real Sentence-Transformer model was not run during development** (the build sandbox could not install PyTorch). Its code path is covered only by a stub. All sample outputs, latencies and the confidence check above use the `lsa` fallback. Please run `examples/parity_check_week3.py` and `examples/validate_confidence.py` (default backend) and `python examples/generate_docs.py` on your machine, then update the numbers in §5/§6.
* Skill matching is exact string matching against a vocabulary mined from job postings (4 983 raw strings, many are long phrases like "ms office (word, powerpoint, …)" that never match a CV). Near-synonyms count as misses — e.g. a student with "communication" is "missing" a JD's "communication skills". A normalisation/alias layer is the highest-value next step.
* Week 5 findings still apply: sparse profiles and adjacent categories (for example SALES/BUSINESS-DEVELOPMENT) can be unreliable. The category signal is deliberately capped at 10% of ranking, and the asset continues to surface uncertainty through warnings and confidence rather than making automated decisions. The Week 4 CSV is the source of truth for the latest proxy-benchmark numbers.
* Scanned/image-only CVs are rejected (no OCR). PDF text order depends on `pypdf`.
* No rate limiting, persistence or per-user auth beyond the optional shared API key; put it behind a gateway for production.
* Confidence is heuristic (see §5); it should drive UI wording ("review this result"), not automated decisions.
