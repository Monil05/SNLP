# Setup and Run Guide

Use the first section for normal runs after setup. The second section is for a
first-time setup on Windows PowerShell. Run commands from the repository root
unless a command changes directory.

## Normal run: three terminals

Keep all three terminals open while using the app.

### Terminal 1 — Group 7 teacher/researcher API

```powershell
cd "SNLP-master\Group_7\SNLP_Project"
uv run python -m src.week6.api
```

Group 7 listens on `http://127.0.0.1:8080`.

### Terminal 2 — Career NLP backend

```powershell
cd "SNLP-master\src"
python -m career_nlp serve --host 127.0.0.1 --port 8000
```

The backend listens on `http://127.0.0.1:8000`.

### Terminal 3 — Frontend and API proxy

```powershell
cd "SNLP-master"
$env:GROUP7_API_URL = "http://127.0.0.1:8080/api"
python frontend/server.py
```

Open `http://127.0.0.1:5500`. Keep the Python frontend server running; it
provides the document extraction route and proxies API requests. If port 5500
is occupied, stop the other server first, or choose another port:

```powershell
$env:FRONTEND_PORT = "5501"
python frontend/server.py
```

Then open `http://127.0.0.1:5501`.

## First-time setup

Install Python 3.12 and `uv`, then open PowerShell in the repository root (the
folder containing `requirements.txt`). The commands below create the Group 8
Python environment, install its requirements and local package, then install
Group 7's separate environment and requirements.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[all]"
python -m pip install uv
cd Group_7\SNLP_Project
uv sync
cd ..\..
```

After setup, run the three terminals above. On the first model use, the
Sentence Transformer may download its model files and needs internet access.
Group 7 may also download model/index assets the first time it starts or
searches.

The document upload controls accept PDF and Word files (`.docx`). Legacy
`.doc` extraction additionally needs LibreOffice or `antiword` installed.
