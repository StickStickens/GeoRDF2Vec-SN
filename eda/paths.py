"""Repository locations shared by all datasets."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "eda" / "figures"
