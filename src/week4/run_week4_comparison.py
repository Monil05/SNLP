"""Run the Week 4 baseline vs improved model comparison.

This launcher performs a dependency preflight before starting the evaluation.
If ``sentence-transformers`` is missing, it installs the packages listed in
the project requirements file using the same Python interpreter that launches
this script. The actual Sentence-Transformer model is still downloaded only
on its first use and is cached by the Week 3 pipeline.
"""

from pathlib import Path
import importlib.util
import subprocess
import sys

BASE_VERSION = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_VERSION / "src" / "week4"
REQUIREMENTS = BASE_VERSION / "requirements.txt"
sys.path.insert(0, str(SRC_DIR))


def ensure_dependencies() -> None:
    """Ensure the Week 4 runtime dependency is installed.

    We deliberately use ``sys.executable`` so the package is installed into
    the same Python environment that runs this project (important in VS Code
    when multiple Python installations/virtual environments exist).
    """
    if importlib.util.find_spec("sentence_transformers") is not None:
        return

    print("Sentence Transformers is not installed in the active Python environment.")
    print("Installing the missing Week 4 package: sentence-transformers")
    print("An internet connection is required if the packages are not already cached.\n")

    if not REQUIREMENTS.exists():
        raise FileNotFoundError(f"Requirements file not found: {REQUIREMENTS}")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "sentence-transformers"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Could not install the Week 4 dependencies.\n"
            "Please activate the correct VS Code virtual environment and run:\n"
            f"  {sys.executable} -m pip install -r \"{REQUIREMENTS}\"\n"
            "Then run this script again. An internet connection may be required."
        ) from exc

    # Refresh the import cache after pip installation.
    importlib.invalidate_caches()
    if importlib.util.find_spec("sentence_transformers") is None:
        raise RuntimeError(
            "sentence-transformers was not available after installation. "
            "Restart the terminal/VS Code interpreter and run the script again."
        )


if __name__ == "__main__":
    ensure_dependencies()
    from evaluation_week4 import main

    main()
