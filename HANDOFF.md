# Group 8 ↔ Group 7 Integration Handoff

## Goal

Connect Group 8's existing CareerMatch UI to Group 7's reusable researcher
recommendation API. Keep Group 8's active `src/career_nlp` Sentence Transformer
pipeline and current matching behavior intact, leave the Week 2/3 baseline
alone, and do not edit Group 7's standalone project.

The integration sends only each Group 8 `recommended_learning` topic to Group 7.
It shows three publication-based researcher results per recommended-learning
topic, with up to two supporting papers per researcher. A separate topic search
shows up to 50 Group 7 matches. Both result areas scroll independently. The
direct search and per-job suggestions remain independent; each has its own
clear control. The teacher publication score and confidence level are displayed
from Group 7's `score` and `confidence.level` response fields. Existing job
scores, matched skills, gaps, and learning lists remain visible.

The app has YouTube-style navigation with **Career Match**, **Search**, and
**Stats** views. Direct searches and per-job suggestions remain independent;
each has its own clear control. Stats shows the Week 4 CSV values hardcoded in
`frontend/app.js`, so it works with any static frontend server.

## Changes made

- `frontend/server.py`: added a `/api/group7/v1/...` proxy.
  `GROUP7_API_URL` defaults to `http://127.0.0.1:8080/api`.
- `frontend/index.html`: added navigable **Career Match**, **Search**, and
  **Stats** sections.
- `frontend/app.js`: added Group 7 topic requests, researcher/evidence cards,
  per-job learning-topic lookup, independent manual search behavior, and
  API error display. Group 8 requests still use the existing `/api/v1/...`
  routes.
- `frontend/styles.css`: styled the new search panel and compact evidence cards.
- `frontend/README.md`: documented both API services and the new researcher
  search behavior.

No changes were made under `Group_7/` or to `src/week2/`, `src/week3/`, or the
Group 8 matching engine.

## Run the integrated demo (Windows PowerShell)

Terminal 1 — Group 7 standalone API:

```powershell
cd "C:\Users\monil\Downloads\SNLP-master\Group_7\SNLP_Project"
uv sync
uv run python -m src.week6.api
```

This listens on `http://127.0.0.1:8080`. Group 7 may download/load the
Sentence-BERT model and publication vectors on the first query.

Terminal 2 — Group 8 Sentence Transformer backend:

```powershell
cd "C:\Users\monil\Downloads\SNLP-master"
python -m career_nlp serve --host 127.0.0.1 --port 8000
```

Use the same Python environment in which Group 8's `career-nlp[all]` package is
installed. The active backend should be configured for the
`sentence-transformers` embedding backend.

Terminal 3 — Group 8 frontend and proxy:

```powershell
cd "C:\Users\monil\Downloads\SNLP-master"
$env:GROUP7_API_URL = "http://127.0.0.1:8080/api"
python frontend/server.py
```

Open `http://127.0.0.1:5500`. Run an analysis with a job description, then
choose “Find researchers for recommended learning” on a job card. The manual
researcher search is available above the job results.

## Contract and limitations

- Group 8 → Group 7 request: `POST /api/v1/recommend` with `query` set to a
  learning topic, `top_k: 3` per job learning topic or `top_k: 50` for direct
  searches, `include_evidence: true`, and `paper_top_k: 100`.
- The frontend sends no student profile, full job description, or gap list to
  Group 7; only the selected learning topic or manual topic query is sent.
- Group 7 returns researchers based on publication matches and evidence. Its
  index does not verify current faculty employment, teaching assignment, or
  mentor availability. Present these as researcher matches, not confirmed
  professors.
- The Group 7 API needs to be running for researcher search. Group 8 matching
  continues to work if it is stopped.

## Verification status / next steps

The Group 7 request and response structures were checked against its local
`src/week6/schemas.py`, `src/week6/api.py`, and `src/week6/asset.py`. The API
endpoint is `POST /api/v1/recommend`, and result records include `name`,
`author_id`, `score`, `confidence`, and `evidence` with paper title, year, and
similarity. JavaScript syntax (`node --check frontend/app.js`) and Python syntax
(`ast.parse` on `frontend/server.py`) both pass. Group 7's nested repository
reports no modified files.

Automated tests and a live three-service browser/API run have not yet been run
after these edits. Next, run the integration using the commands above, exercise
both automatic learning-topic lookup and manual topic search, then run the
relevant existing Group 8 frontend/API checks if desired. If Group 7 runs on a
different address, set `GROUP7_API_URL` to its API prefix (for example,
`http://127.0.0.1:8081/api`) before starting `frontend/server.py`.
