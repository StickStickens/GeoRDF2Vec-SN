# Preprocessing: what we changed and why

**Principle.** Everything our experiments do not test stays as in the authors' pipeline, so that differences in results come from flooding and walks only. We change only what the EDA showed to be a bug. Evidence is in [dmg777k_preprocessing.ipynb](dmg777k_preprocessing.ipynb) and [bgs_preprocessing.ipynb](bgs_preprocessing.ipynb).

## Changes to the authors' pipeline

| Change | Why | dmg777k | BGS+ |
|---|---|---|---|
| A node is identified by its text **and** kind (entity, or the literal's datatype or language tag) | Identifying by text alone merges an entity with a literal of the same text, which creates self-loops, and merges different values, such as a monument number and a population | 9,075 wrong merges and 8,396 self-loops removed | 59,870 wrong merges and 58,689 self-loops removed |
| Placeholder literals ("", "Not Entered", "Unknown", …) and empty geometries (`POLYGON EMPTY`) are removed | They mean "no value", yet become hubs that link unrelated entities | 204 triples (the same ones the authors' `dropna()` removes) | 33,687 triples |
| A geometry is any WKT-typed literal, in any letter case | The authors' case-sensitive expression misses `Point(...)` | 21,208 geometries instead of 12,851 | 48,603 (no change) |
| WKT keywords are written in capitals (`Point(` becomes `POINT(`) | WKT ignores letter case, but code that searches node text does not; no geometry occurs in both spellings, so nothing merges | 8,357 literals rewritten | none needed |
| Coordinates are (longitude, latitude) throughout | The authors pass them to geopy swapped | Amsterdam–Utrecht 34.6 km instead of 39.4 km | same |

The authors' steps are otherwise kept: dmg777k's RD New outlines are converted to WGS84 as in `dmg_data_prep.py`, and BGS+ is prepared the same way by `data/bgs_data_prep.py`. As a result, our dmg777k graph has 341,269 nodes instead of the paper's 332,194, with the same 776,920 triples. Reproducing the paper's numbers therefore uses the unchanged code in `original/`.

## Outputs (`data/<dataset>/processed/`)

`triples.parquet` (node keys, object kind), `geometries.parquet` (WKT and centre of each geometry), `labels.csv` (node, class, split), `relations.csv` (one row per relation: size, entity or literal objects, spatial spread) and `hubs.csv` (the 50 nodes with the most edges). Label checks pass for both datasets: no node is labelled twice, all labelled nodes are in the graph, and the target relation is absent.

## Left to the experiments

Walk direction and start nodes, which relations pass on locations during flooding, how to treat hubs, and the edge weights. The relation table and the hub list are the facts these decisions rest on; the decisions are made, and compared, in `experiments/`.
