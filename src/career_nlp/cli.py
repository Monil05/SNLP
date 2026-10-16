"""Command-line interface:  python -m career_nlp <command> ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import AssetConfig
from .engine import CareerNLP
from .errors import CareerNLPError


def _text(value: Optional[str], file: Optional[str], name: str) -> Optional[str]:
    if file:
        return Path(file).read_text(encoding="utf-8")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="career_nlp", description="Career NLP asset CLI (JSON output)")
    p.add_argument("--backend", choices=["sentence-transformers", "lsa", "auto"],
                   help="Embedding backend (default: from env / sentence-transformers)")
    p.add_argument("--jobs", help="Path to the jobs CSV (default: base version/data/cleaned_jobs.csv)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def student(sp):
        sp.add_argument("--student", help="Student/CV text")
        sp.add_argument("--student-file", help="Read student text from a file")

    sp = sub.add_parser("match", help="JD provided"); student(sp)
    sp.add_argument("--job"); sp.add_argument("--job-file"); sp.add_argument("--details", action="store_true")
    sp = sub.add_parser("recommend", help="JD not provided"); student(sp)
    sp.add_argument("--top-k", type=int, default=3); sp.add_argument("--details", action="store_true")
    sp = sub.add_parser("extract"); sp.add_argument("--text"); sp.add_argument("--file")
    sp = sub.add_parser("classify"); sp.add_argument("--text"); sp.add_argument("--file")
    sp = sub.add_parser("cv", help="Analyse a PDF/DOCX CV"); sp.add_argument("path")
    sub.add_parser("info")
    sp = sub.add_parser("serve", help="Run the REST API")
    sp.add_argument("--host", default="127.0.0.1"); sp.add_argument("--port", type=int, default=8000)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = AssetConfig.from_env()
    overrides = {}
    if args.backend:
        overrides["embedding_backend"] = args.backend
    if args.jobs:
        overrides["jobs_path"] = Path(args.jobs)
    config = config.with_overrides(**overrides) if overrides else config

    if args.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(config=config), host=args.host, port=args.port)
        return 0

    nlp = CareerNLP(config)
    try:
        if args.cmd == "match":
            s, j = _text(args.student, args.student_file, "student"), _text(args.job, args.job_file, "job")
            result = nlp.match(s or "", j or "", args.details)
        elif args.cmd == "recommend":
            result = nlp.recommend(_text(args.student, args.student_file, "student") or "", args.top_k, args.details)
        elif args.cmd == "extract":
            result = nlp.extract(_text(args.text, args.file, "text") or "")
        elif args.cmd == "classify":
            result = nlp.classify(_text(args.text, args.file, "text") or "")
        elif args.cmd == "cv":
            result = nlp.analyze_cv(args.path)
        else:
            result = nlp.info()
    except CareerNLPError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message, "details": exc.details}}, indent=2),
              file=sys.stderr)
        return 2
    print(result.model_dump_json(indent=2, exclude_none=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
