# Exploratory analysis (Milestone 2)

| File | Contents |
|---|---|
| [dmg777k_eda.ipynb](dmg777k_eda.ipynb), [bgs_eda.ipynb](bgs_eda.ipynb) | The analysis of each dataset |
| [dmg777k_findings.md](dmg777k_findings.md), [bgs_findings.md](bgs_findings.md) | Main outcomes of each notebook |
| [eda_report.md](eda_report.md), [eda_report.pdf](eda_report.pdf) | The submitted two-page report, as handed in (its paths refer to the earlier layout) |
| [figures/](figures) | Figures written by the notebooks, as PDF and PNG |
| `graph.py`, `spatial.py`, `plots.py`, `paths.py` | Helper code: the graph built as the authors' code builds it, a re-implementation of their flooding and edge weighting with switches for our fixes, spatial statistics, walk simulation and plotting |
| `dmg777k/`, `bgs/` | Dataset loaders |

To reproduce, prepare the data as described in [data/README.md](../data/README.md), then run from the repository root:
```
uv run jupyter nbconvert --to notebook --execute --inplace eda/dmg777k_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace eda/bgs_eda.ipynb
```
This rewrites the figures in [figures/](figures); the BGS+ ones have a `bgs_` prefix.
