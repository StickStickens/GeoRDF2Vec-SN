"""Repository locations shared by all datasets."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
