# Experiments (Milestone 4+)

| File | Contents |
|---|---|
| [evaluation_plan.md](evaluation_plan.md) | Learning setting, splits, metrics and compared methods |
| [run.py](run.py) | Trains one configuration for some seeds and scores it on validation data |
| `results/<dataset>/` | One JSON per run: configuration, seed, machine, timings, validation scores |

Input data: `data/<dataset>/processed/`, written by [preprocessing](../preprocessing/README.md).

Milestone 4 is due on 7 December. We aim to finish **all** experiments in it, not only a first iteration.

## Left to the experiments

Preprocessing makes no modelling decisions. Decided here:

- walk direction and start nodes,
- which relations pass on locations during flooding, and what to do with hubs (`relations.csv` and `hubs.csv` describe them),
- flooding and edge weights.

## What the authors' code does, and what we change

Found in the EDA and in a reading of `original/geo_rdf2vec.py`. Keep `original/` unchanged for reproducing the paper. Our pipeline is new code in `experiments/`.

**Bugs**

1. **Word2vec trains on nothing.** In `kg_training`, `corpus` is an iterator that `build_vocab` uses up, so `train` sees 0 words. `corpus = list(...)` fixes it, but holding every walk in memory does not fit in 16 GB once walks go both ways. We write walks to a text file and train with gensim's `corpus_file`.
2. **Longitude and latitude are swapped.** `extract_point_coordinates` returns (lon, lat), but geopy expects (lat, lon). Amsterdam–Utrecht comes out as 39.4 km instead of 34.6 km.
3. **Geometry detection is case-sensitive** and misses `Point(...)` on the raw dmg777k data. No change needed: our preprocessing writes WKT keywords in capitals.
4. **The min-max edge weights are wrong and too slow.** Each edge is weighted once from each end, and the last write wins. The loop filters the whole edge table for every node (about 341k × 777k operations). The EDA has a vectorised version (`repo_minmax_weights` in [eda/spatial.py](../eda/spatial.py)).

**Walks**

- Walks use igraph's `random_walk`, which picks the next edge in proportion to its weight. It follows outgoing edges only and can return along the edge it came from.
- Every node, literals included, starts 100 walks of depth 8. These values are hard-coded and override the constructor arguments; `flood_direction` and `walk_extraction` are unused.
- Degree down-weighting fits this scheme (multiply edge weights). Inverse edges without backtracking, and a separate number of walks for geo-located nodes, need our own walker. `simulate_walks` in [eda/spatial.py](../eda/spatial.py) is a starting point. A vectorised walker generates all walks for one dataset in seconds, so word2vec is the only real cost.

**Classifier**

`original/modeling/dmg_hpt.py` tunes XGBoost with 60 rounds of Bayesian optimisation per embedding, optimises MCC and reports micro F1. See [Classifier](#classifier) for what we use instead.

## Experiments

Each experiment changes one factor against a fixed base. Factors that help are then combined. Embeddings never see labels, so a configuration costs one word2vec training per seed; the BGS+ cross-validation only repeats the classifier.

**Base, "GeoRDF2Vec fixed":** our preprocessed graph, bugs 1, 2 and 4 fixed, the authors' flooding and min-max weights, forward walks, 100 walks of depth 8 per node.

| # | Experiment | Variants | Configs |
|---|---|---|---|
| E0 | Baselines | majority class and kNN on coordinates (no embedding); RDF2Vec with uniform walks, forward as the authors walk and with each of E1's two walk rules (this separates the effect of direction from that of geography); GeoRDF2Vec as published, with only bug 1 fixed | 4 |
| — | Base | as above | 1 |
| E1 | **Walk direction** | walks start at entities, follow edges both ways and never step straight back. (a) Inverse edges only between entities, so a walk ends at the first literal. (b) Inverse edges for every edge, so walks pass through literals; a walk steps back only at a dead end, such as a literal used by one entity | 2 |
| E2 | Relation-aware flooding | only relations whose objects lie close together (from `relations.csv`) pass on locations; also checked without labels, see below | 1 |
| E3 | **Hub removal** | remove the top k hubs, k ∈ {10, 50} | 2 |
| E4 | **Degree down-weighting** | edge weight × degree(target)^−α, α ∈ {0.5, 1} | 2 |
| E5 | **Language tags** | strip language tags, so `"Amsterdam"@nl` and `"Amsterdam"@en` become one node | 1 |
| E6 | **Walks per geo-located node** | {50, 200} walks for nodes with a location, against 100 in the base | 2 |
| E7 | Adaptive walk length | the four rules below, on walks of type E1 (b) | 4 |
| E8 | Best combination | winners of E1–E7 together; E3 with E4, since both target hubs; spatial block cross-validation as a robustness check | ~2 |

About 21 configurations × 5 seeds × 2 datasets ≈ **210 word2vec trainings**.

**Order.** E1 runs first. Forward walks end after about 1.7 steps, so E6 and E7 only mean something once walks can go both ways. E2–E6 then build on the walk type E1 picks; E7 always uses E1 (b). E2's label-free check can run before any training.

**Flooding check (E2, no labels).** Hide known geometries (20% of dmg777k monuments; all 3,290 BGS+ stratotypes), re-run flooding, and measure the median and 90th-percentile distance error, the share placed in the correct municipality (dmg777k), and the share placed at the national centroid.

### Adaptive walk length (E7)

A length rule needs walks that get long. Measured on 200k walks per dataset (20k entity starts, depth 8):

| Walk type | dmg777k: mean steps, reach 8 | BGS+: mean steps, reach 8 |
|---|---|---|
| Forward | 1.0, 0% | 1.0, 0% |
| E1 (a): end at the first literal | 2.3, 1.5% | 3.4, 11.6% |
| Through literals, never step back (walks stop at literals used by one entity) | 3.8, 17.5% | 4.1, 21.9% |
| E1 (b): through literals, step back only at a dead end | 8.0, 100% | 8.0, 100% |

Only E1 (b) walks get long, as in the EDA, where they kept drifting away from the start (16 → 87 km on dmg777k). Passing through shared literals is also where walks jump far: monuments that share a build year are 66–73 km apart. E7 therefore runs on E1 (b).

All four rules are tested:

1. **Distance cap:** stop once the walk is more than *d* km from its start node's location. This is the main candidate.
2. **Distance-based stopping probability:** at each step, stop with probability 1 − exp(−distance / *d*). This is a soft version of 1 without a hard edge.
3. **Stop at location-less nodes:** stop on entering a node whose location is unreliable, such as one flooded to the national centroid. 91–97% of both-way BGS+ walks touch such a hub.
4. **Distance budget:** stop once the summed length of the steps exceeds *d* km. Unlike 1, this also stops walks that wander back and forth near the start.

*d* is set per dataset from the EDA distances, since BGS+ covers a larger area than dmg777k.

### Classifier

**Logistic regression** (standardised vectors, C tuned on validation over {0.001, 0.01, 0.1, 1, 10}) for every run. It is as accurate as tuned XGBoost here and much faster.

Measured on one dmg777k embedding (walks both ways from entities, 100 × depth 8, no spatial weights), trained on the training split, scored on validation:

| Classifier | Macro F1 | Accuracy | Tuning time |
|---|---|---|---|
| Majority class | — | 0.478 | — |
| Logistic regression, 5 values of C | **0.702** | **0.747** | 29 s |
| XGBoost, default settings | 0.636 | 0.712 | 9 s |
| XGBoost, 20-config random search | 0.697 | 0.735 | 553 s |

A linear model also suits BGS+, with only 117 labelled units per cross-validation. XGBoost with the authors' Bayesian tuning is kept only for the paper-reproduction row. This is one embedding on one dataset: repeat the comparison on the final configuration of each dataset.

## Run time and machines

Measured on this PC (Ryzen 5 3600X, 6 cores / 12 threads), word2vec with 200 dimensions, window 5, 10 epochs. Times are extrapolated from 5% of the start nodes; the dmg777k both-way run was also timed in full.

| Walks | dmg777k | BGS+ |
|---|---|---|
| Forward, from all nodes (authors) | 23 min (53 min if one-token walks are kept) | about 20 min, estimated from dmg777k (45 min if kept, measured) |
| Both ways through literals, from entities (E1 (b)) | 15 min (full run; 17 min extrapolated) | 11 min |
| E1 (a), ending at the first literal | about 8 min, estimated from the token count | about 6 min, estimated |

- **Skip one-token walks.** Most forward walks start at a literal and stop at once. They give word2vec no training pairs but cost half its time, so we drop them; the vectors of the labelled nodes are unaffected.
- **Total compute.** 210 runs × about 15 min ≈ 53 h on the PC, an upper bound since E1 (a) walks are shorter than in the benchmark; E1 (b) walks are as long as there. With the MacBook Air M1 as a second machine that is roughly 1–1.5 days of running, done overnight. The M1 is not measured yet: run the benchmark there first, since a fanless laptop slows down under hours of full load.
- **Splitting the work.** Run all seeds of a configuration on the same machine, and log the machine with each run.
- **Fallback.** If runs take longer than this, compare variants with 3 seeds and use 5 only for the final table.

## Running on two machines

Every (dataset, configuration, seed) is an independent run. The standing split is **dmg777k on the PC, BGS+ on the MacBook**: BGS+ runs are shorter, which suits the slower, fanless laptop, and each machine only needs one dataset.

- The PC needs `data/kgbench_dmg777k/processed/`. The Mac needs only `data/bgs/processed/` (74 MB), copied from the PC, so it skips the BGS+ download and preparation.
- Run all seeds of a configuration on one machine. Each result records the machine name.
- A run whose result file exists is skipped, so after an interruption re-run the same command.
- To merge, copy `experiments/results/bgs/` from the Mac into the PC's checkout.
- On the Mac, prefix commands with `caffeinate -i` and keep it plugged in with the lid open, or macOS sleeps and the run stops.

## Schedule

| Dates | Work |
|---|---|
| 24–26 Nov | Pipeline: one config → graph variant → flooding → edge weights → walker → walks file → word2vec → classifier → one results row (config, seed, machine, metrics). Check: "GeoRDF2Vec as published" lands near the paper's Table 2. Benchmark the M1. In parallel: the E2 flooding check, and a preprocessing variant without language tags (E5). |
| 27 Nov – 1 Dec | E0–E7 |
| 2–3 Dec | E8 and the spatial block cross-validation |
| 4–5 Dec | Results tables and write-up |
| 6–7 Dec | Buffer |

The pipeline has three parts that can be built in parallel once the config format is agreed:

- **(A)** walker and word2vec;
- **(B)** graph variants, flooding, edge weights and the flooding check;
- **(C)** evaluation: splits, BGS+ repeated cross-validation, metrics, baselines and tables.
