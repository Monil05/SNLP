# CareerMatch Frontend

A small, dependency-free HTML/CSS/JavaScript frontend for the `career_nlp` API. It decides which matching endpoint to call based on how many job descriptions are supplied:

- **Student profile only** → `POST /v1/recommend`
- **Student profile + one job description** → `POST /v1/match`
- **Student profile + multiple job descriptions** → `POST /v1/compare-jobs`

Only the student profile is required. Job descriptions are optional.

The student profile and each job description also have a **Upload** control.
PDF, `.docx`, and legacy `.doc` files up to 5 MB are accepted. Extracted text
is appended to that field so it can be reviewed or edited before submitting.
PDF extraction uses `pypdf`; `.docx` extraction uses `python-docx` (both are in
the project requirements). Legacy `.doc` files need LibreOffice or `antiword`
installed on the machine running the frontend server; saving as `.docx` also
works without that extra system tool. Scanned PDFs need OCR and cannot be read
as selectable text.

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

Uploads first use the Python frontend server. When the page is hosted by VS Code
**Go Live** or another static server, uploads fall back to the Career NLP API on
port 8000; keep that backend running and open the page from `localhost` or
`127.0.0.1` so its local CORS rule applies. The Python frontend server is still
needed for legacy `.doc` extraction. If port 5500 is occupied, set
`$env:FRONTEND_PORT = "5501"` before running `python frontend/server.py`, then
open `http://127.0.0.1:5501`.

The frontend server proxies Group 8 requests from `/api/...` to `http://127.0.0.1:8000`. Researcher searches use `/api/group7/...` and are proxied to Group 7's standalone API at `http://127.0.0.1:8080/api` by default. The browser does not make cross-origin API requests. Start both APIs in their own terminals.

Use **Load example** to populate a sample student and three job descriptions. You can remove all job descriptions to return to recommendation mode.

The app sidebar has three sections: **Career Match** for the existing student/JD scenarios and per-job teacher lookup, **Search** for direct Group 7 topic search, and **Stats** for the saved baseline/Core evaluation. Direct searches and per-job results are independent; each has its own clear control.

Career Match can search Group 7 for three publication-based researcher matches per recommended-learning topic. Search shows up to 50 Group 7 matches. Both result lists scroll within their own sections. Stats displays the Precision@5, Recall@5, and MRR values copied from `test/week4_model_comparison.csv` in `frontend/app.js`.

For the full collaboration, start Group 7's standalone API in its own terminal from the Group 7 project folder:

```powershell
cd "Group_7\SNLP_Project"
uv run python -m src.week6.api
```

It listens on port `8080` by default. Start Group 8's backend on `8000` and frontend on `5500` as shown above. Use the absolute Group 7 folder path if your terminal starts from somewhere else.

To use another Group 7 API address, set `GROUP7_API_URL` before starting the frontend server, for example:

```powershell
$env:GROUP7_API_URL = "http://127.0.0.1:8080/api"
python frontend/server.py
```

Group 7's names and evidence refer to researchers/publication authors in its index. They do not establish that someone is a current professor, teaches a specific course, or is available to mentor.
