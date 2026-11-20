# Experiments (Milestone 4+)

| File | Contents |
|---|---|
| [evaluation_plan.md](evaluation_plan.md) | Learning setting, splits, metrics and compared methods |

Input data: `data/<dataset>/processed/`, written by [preprocessing](../preprocessing/README.md).

## Left to the experiments

Preprocessing makes no modelling decisions. Decided here:

- walk direction and start nodes,
- which relations pass on locations during flooding, and what to do with hubs (`relations.csv` and `hubs.csv` describe them),
- flooding and edge weights.

## To fix in the authors' code (`original/geo_rdf2vec.py`)

Found in the EDA. Keep `original/` unchanged for reproducing the paper; fix these in our copy.

1. **Word2vec trains on nothing.** In `kg_training`, `corpus` is an iterator that `build_vocab` uses up, so `train` sees 0 words. Fix: `corpus = list(...)`.
2. **Longitude and latitude are swapped.** `extract_point_coordinates` returns (lon, lat), but geopy expects (lat, lon). Amsterdam–Utrecht comes out as 39.4 km instead of 34.6 km.
3. **Geometry detection is case-sensitive** and misses `Point(...)` on the raw dmg777k data. No change needed: our preprocessing writes WKT keywords in capitals.

## Oliwier's suggest for the walks

- Walks start only from entities.
- A walk ends at the first literal it reaches.
- Edges between entities may be followed in both directions; edges to literals only forwards.
- Cycles are handled by the walker (for example, no immediate return along the same edge).
- One run keeps the authors' forward-only walks, to reproduce the paper.
