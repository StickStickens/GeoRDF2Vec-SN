"""Shared chart style and figure export."""

from pathlib import Path

import matplotlib.pyplot as plt

from .paths import FIGURES_DIR

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b8b7b0"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CLASS_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": NEUTRAL,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "figure.constrained_layout.use": True,
        }
    )


def save(figure: plt.Figure, name: str, directory: Path = FIGURES_DIR) -> None:
    """Export a figure as PDF and PNG for the report."""
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.pdf")
    figure.savefig(directory / f"{name}.png")
