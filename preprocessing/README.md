# Preprocessing (Milestone 3)

| File | Contents |
|---|---|
| [dmg777k_preprocessing.ipynb](dmg777k_preprocessing.ipynb), [bgs_preprocessing.ipynb](bgs_preprocessing.ipynb) | Each step with its numbers before and after, and the checks that the problems from the EDA are gone |
| [preprocessing.md](preprocessing.md) | Summary: what we changed relative to the authors' pipeline, and why |
| `load.py` | Reads each dataset into one triple table that keeps every object's type |
| `graph.py` | Node identity by text and kind; removal of placeholder values |
| `geometry.py` | WKT geometries, their centres, and geodesic distances |
| `catalog.py` | The relation table and the hub list |
| `labels.py` | Split counts and label leakage checks |
| `pipeline.py` | Runs all steps for one dataset and writes the outputs |

Prepare the data as described in [data/README.md](../data/README.md), then run from the repository root either the notebooks:
```
uv run jupyter nbconvert --to notebook --execute --inplace preprocessing/dmg777k_preprocessing.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace preprocessing/bgs_preprocessing.ipynb
```
or the same steps without them:
```
uv run python -m preprocessing.pipeline dmg777k bgs
```
Both write to `data/kgbench_dmg777k/processed/` and `data/bgs/processed/`. The experiments read these files, or import `preprocessing.pipeline.run` directly.
