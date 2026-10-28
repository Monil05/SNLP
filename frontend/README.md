# CareerMatch Frontend

A small, dependency-free HTML/CSS/JavaScript frontend for the `career_nlp` API. It decides which matching endpoint to call based on how many job descriptions are supplied:

- **Student profile only** → `POST /v1/recommend`
- **Student profile + one job description** → `POST /v1/match`
- **Student profile + multiple job descriptions** → `POST /v1/compare-jobs`

Only the student profile is required. Job descriptions are optional.

Results are intentionally concise: each role shows its final job-match percentage, matched skills, required gaps, and recommended learning. The underlying API still returns the detailed fields needed to populate these sections, but score breakdowns and raw JSON are not shown in the interface.

## Run it

Open two terminals at the project root.

**Terminal 1 — backend**

```powershell
python -m career_nlp serve --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend**

```powershell
python frontend/server.py
```

Open http://127.0.0.1:5500 in your browser.

The frontend server proxies requests from `/api/...` to `http://127.0.0.1:8000`, so the browser does not make cross-origin API requests and no CORS environment variable is needed for this local setup. The backend must still be running in its own terminal.

Use **Load example** to populate a sample student and three job descriptions. You can remove all job descriptions to return to recommendation mode.
