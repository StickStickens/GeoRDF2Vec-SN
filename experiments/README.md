# Experiments (Milestone 4+)

| File | Contents |
|---|---|
| [evaluation_plan.md](evaluation_plan.md) | Learning setting, splits, metrics and compared methods |
| [run.py](run.py) | Trains configurations for some seeds and scores them on validation data; `--list` shows the configurations |
| [geo.py](geo.py) | Relation-aware relation choice, flooding, spatial edge weights |
| [flood_check.py](flood_check.py) | E2 without labels: hide known locations, flood, measure the error |
| `results/<dataset>/` | One JSON per run (configuration, seed, machine, walk statistics, timings, validation scores), and `flood_check.csv` |

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
4. **The min-max edge weights are wrong and too slow.** Each edge is weighted once from each end, and the last write wins. The loop filters the whole edge table for every node (about 341k × 777k operations). Our fix (`minmax_weights` in [geo.py](geo.py)) normalises at each node over the edges a walk can take next, so an edge has one weight per direction it is walked in. The published variant keeps the authors' behaviour (`authors_minmax_weights`, identical to the EDA's checked `repo_minmax_weights` on all 916k BGS+ edges).

**Walks**

- Walks use igraph's `random_walk`, which picks the next edge in proportion to its weight. It follows outgoing edges only and can return along the edge it came from.
- Every node, literals included, starts 100 walks of depth 8. These values are hard-coded and override the constructor arguments; `flood_direction` and `walk_extraction` are unused.
- Degree down-weighting fits this scheme (multiply edge weights). Inverse edges without backtracking, and a separate number of walks for geo-located nodes, need our own walker. `simulate_walks` in [eda/spatial.py](../eda/spatial.py) is a starting point. A vectorised walker generates all walks for one dataset in seconds, so word2vec is the only real cost.

**Classifier**

`original/modeling/dmg_hpt.py` tunes XGBoost with 60 rounds of Bayesian optimisation per embedding, optimises MCC and reports micro F1. See [Classifier](#classifier) for what we use instead.

## Experiments

Each experiment changes one factor against a fixed base. Factors that help are then combined. Embeddings never see labels, so a configuration costs one word2vec training per seed; the BGS+ cross-validation only repeats the classifier.

**Base, "GeoRDF2Vec fixed":** our preprocessed graph, bugs 1, 2 and 4 fixed, the authors' flooding and min-max weights, forward walks, 100 walks of depth 8 per node.

E0 showed walks through literals (E1 (b)) to be best on both datasets without geography (macro F1 0.677 vs 0.627 forward on dmg777k, 0.761 vs 0.548 on BGS+), so E2–E7 are configured on E1 (b) walks with geography. The E1 runs with geography (`geo_forward`, `geo_both`, `geo_through_literals`) check that this holds.

| # | Experiment | Variants | Configs |
|---|---|---|---|
| E0 | Baselines | majority class and kNN on coordinates (no embedding); RDF2Vec with uniform walks, forward as the authors walk and with each of E1's two walk rules (this separates the effect of direction from that of geography); GeoRDF2Vec as published, with only bug 1 fixed | 4 |
| — | Base | as above | 1 |
| E1 | **Walk direction** | walks start at entities, follow edges both ways and never step straight back. (a) Inverse edges only between entities, so a walk ends at the first literal. (b) Inverse edges for every edge, so walks pass through literals; a walk steps back only at a dead end, such as a literal used by one entity | 2 |
| E2 | Relation-aware flooding | only relations whose objects lie close together (from `relations.csv`) pass on locations; also checked without labels, see below | 1 |
| E3 | **Hub removal** | remove the top k hubs, k ∈ {10, 50} | 2 |
| E4 | **Degree down-weighting** | edge weight × degree(target)^−α, α ∈ {0.5, 1} | 2 |
| E5 | **Language tags** | strip language tags, so literals with identical text become one node. Translations (`"kerk"@nl`, `"church"@en`) are not merged. Both datasets are almost monolingual, so the merges join tagged and untagged copies of the same string: on dmg777k 266 groups, mostly municipality names such as `"Aalsmeer"@nl` and `"Aalsmeer"`, only 5 of them across two languages; on BGS+ 20 groups of codes. 753 and 4,034 triples are affected, so a small effect is expected | 1 |
| E6 | **Walks per geo-located node** | {50, 200} walks for nodes with a location, against 100 in the base | 2 |
| E7 | Adaptive walk length | the four rules below, on walks of type E1 (b) | 4 |
| E8 | Best combination | winners of E1–E7 together; E3 with E4, since both target hubs; spatial block cross-validation as a robustness check | ~2 |

About 21 configurations × 5 seeds × 2 datasets ≈ **210 word2vec trainings**.

**Order.** E1 runs first. Forward walks end after about 1.7 steps, so E6 and E7 only mean something once walks can go both ways. E2–E6 then build on the walk type E1 picks; E7 always uses E1 (b). E2's label-free check can run before any training.

### Configurations

| # | Configurations (`run.py`) |
|---|---|
| E0 | `rdf2vec_forward`, `rdf2vec_both`, `rdf2vec_through_literals` (done), `geo_published` |
| Base, E1 | `geo_forward`, `geo_both`, `geo_through_literals` |
| E2 | `geo_relation_aware` |
| E3 | `geo_hubs10`, `geo_hubs50` |
| E4 | `geo_degree05`, `geo_degree1` |
| E5 | `geo_nolang` |
| E6 | `geo_walks50`, `geo_walks200` |
| E7 | `geo_len_cap`, `geo_len_soft`, `geo_len_budget`, `geo_len_unreliable` |

Choices made in the code, open to change:

- **Relation-aware flooding (E2).** A relation passes on locations if the located subjects sharing one of its objects lie within 0.4 × the dataset's baseline spread of their group centre (30 km on dmg777k, 99 km on BGS+), or, where too few subjects are located to measure this, if its objects are shared by at most 20 subjects on average. WKT relations always pass. 0.4 is a hand-picked value: it reproduces the EDA's hand-made BGS+ list (0.5 keeps ages, colours and map codes at 106–123 km; 0.33 drops `hasLexicon` at 85 km, which links the map to the rock units). It uses no labels, and the sensitivity check below shows that every value from 0.35 to 0.42 gives the same relations on both datasets, with no worse flooding anywhere from 0.05 to 0.45. This keeps 40 of 60 relations on dmg777k and 70 of 103 on BGS+. 18% of dmg777k nodes, reachable only through dropped relations, get no location; edges at them count as distance 0, as in the authors' code.
- **Unreliable node (E7, rule 3):** flooding gave it more than half of all geometries. That is 17% of dmg777k nodes and 53% of BGS+ nodes with the authors' flooding (EDA: 18.5% and 50.3% inherit all of them), and none after relation-aware flooding. Removing hubs helps less on BGS+: 10% of its nodes stay unreliable after removing the top 10 hubs, and none after removing the top 50.
- **Length rules (E7):** *d* = 0.4 × the baseline spread, the same 30 km / 99 km. A walk ends before a step that breaks rules 1, 2 and 4, and right after stepping onto an unreliable node for rule 3. Walks then average about 4–5 steps instead of 8.
- **Geo-located node (E6):** an entity with its own geometry (31,169 on dmg777k, 22,784 on BGS+). After flooding every node has a location, so "located" has to mean its own.
- **Hub removal (E3)** removes the k highest-degree nodes, skipping any labelled node and any hub whose removal would leave a labelled node without edges (`select_hubs`; skipped hubs are listed in the result as `hubs_skipped`). The only node skipped is `monument/8336` on dmg777k (hub 46, degree 1,105), which is itself a labelled monument; the top-50 list then takes hub 51 instead. Without this rule the monument itself would be removed, and it would get a zero vector. No other labelled node loses its edges, and every labelled node stays in the largest connected component for k ≤ 50 on both datasets. The nodes that do lose all their edges were attached only to a hub, e.g. on dmg777k 57,834 images and geometries whose only edge is `rdf:type` to `Afbeelding` or `Geometry`.

### Flooding check (E2, no labels)

Hide known locations in 5 folds, flood again, and measure where flooding puts them ([flood_check.py](flood_check.py)): on dmg777k the `hasGeometry` links of the 10,303 entities with their own geometry, on BGS+ the `hasStratotype` links of the 3,290 units with a stratotype (truth: the mean location of their stratotypes).

| Dataset | Flooding | Median error | 90th percentile | At the national centroid |
|---|---|---|---|---|
| dmg777k | all relations (authors) | 2.3 km | 7.4 km | 0% |
| dmg777k | relation-aware | 2.2 km | 6.5 km | 0% |
| BGS+ | all relations (authors) | 238 km | 480 km | 98% |
| BGS+ | relation-aware | 32 km | 224 km | 0% |

On BGS+, relation-aware flooding removes the national-centroid problem for these units, as in the EDA (389 → 59 km on its 97 labelled units). On dmg777k it changes little: hidden monuments still get their location from their address and city. The share placed in the correct municipality is not measured yet.

**Sensitivity to the 0.4 spread fraction** (`flood_check.py --sweep`, written to `flood_sweep.csv`; one row per distinct set of passing relations, so each fraction stands for a range):

| Dataset | Spread fraction | Median error | 90th percentile | At the national centroid | Nodes with a location |
|---|---|---|---|---|---|
| dmg777k | 0.05–0.8 (incl. 0.4) | 2.2 km | 6.5 km | 0% | 81.5–82.1% |
| dmg777k | 1.0 | 2.3 km | 7.3 km | 0% | 82.5% |
| BGS+ | 0.05–0.34 (without `hasLexicon`) | 30 km | 234 km | 0% | 80.9% |
| BGS+ | 0.35–0.42 (incl. 0.4) | 32 km | 224 km | 0% | 98.0% |
| BGS+ | 0.45 (adds map codes) | 32 km | 224 km | 0% | — |
| BGS+ | 0.5–0.8 (adds colours, ages, rock class) | 99 km | 353 km | 0% | 98.1% |
| BGS+ | 1.0 | 105 km | 371 km | 13% | 98.2% |

The labelled nodes get a location at every fraction. On dmg777k the fraction hardly matters. On BGS+ there is a cliff between 0.45 and 0.5, where colours (0.45), ages (0.47–0.48) and the rock class (0.49) start to pass: the median error triples, and 3.6% of nodes (8% at 1.0) inherit more than half of all geometries again. Values from 0.35 to 0.45 are equally good. Below 0.34 the error stays the same, but `hasLexicon` drops out and 17% of BGS+ nodes get no location.

`isStratotypeFor` points to separate `…/ref/Lexicon/NamedRockUnit/…` nodes, not to the `…/id/…` units that carry the labels, so only `hasStratotype` links are hidden.

**Spatial weights do bias the walks.** On BGS+ walks of type E1 (b), the median distance from the start after 8 steps is 2 km with our min-max weights against 61 km with uniform walks.

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
| Both ways through literals, from entities (E1 (b)) | 18 min (measured on 5 seeds) | 17 min on the M1 |
| E1 (a), ending at the first literal | 20 min (measured: short walks are slow per token) | 16 min on the M1 |

- **Skip one-token walks.** Most forward walks start at a literal and stop at once. They give word2vec no training pairs but cost half its time, so we drop them; the vectors of the labelled nodes are unaffected.
- **Total compute.** About 20 min per run on both machines. The 16 geographic configurations are 80 runs per dataset, about 27 h on each machine in parallel. Flooding takes 7–60 s per variant and is cached in `data/<dataset>/runs/_flood/`.
- **Splitting the work.** Run all seeds of a configuration on the same machine, and log the machine with each run.
- **Fallback.** If runs take longer than this, compare variants with 3 seeds and use 5 only for the final table.

## Running on two machines

Every (dataset, configuration, seed) is an independent run. The standing split is **dmg777k on the PC, BGS+ on the MacBook**: BGS+ runs are shorter, which suits the slower, fanless laptop, and each machine only needs one dataset.

- The PC needs `data/kgbench_dmg777k/processed/` and `full_triple_set.csv`. The Mac needs `data/bgs/processed/` (74 MB), `data/bgs/full_triple_set.csv` (189 MB, read by `geo_published`) and, to skip recomputing it, the flooding cache `data/bgs/runs/_flood/` (45 MB), all copied from the PC.
- Run all seeds of a configuration on one machine. Each result records the machine name.
- A run whose result file exists is skipped, so after an interruption re-run the same command.
- To merge, copy `experiments/results/bgs/` from the Mac into the PC's checkout.
- On the Mac, prefix commands with `caffeinate -i` and keep it plugged in with the lid open, or macOS sleeps and the run stops.

## Schedule

| Dates | Work |
|---|---|
| 24–25 Nov | Pipeline and geography (done); E0 without geography (done); E2 flooding check (done) |
| 27 Nov – 1 Dec | E0–E7 |
| 2–3 Dec | E8 and the spatial block cross-validation |
| 4–5 Dec | Results tables and write-up |
| 6–7 Dec | Buffer |

The pipeline has three parts that can be built in parallel once the config format is agreed:

- **(A)** walker and word2vec;
- **(B)** graph variants, flooding, edge weights and the flooding check;
- **(C)** evaluation: splits, BGS+ repeated cross-validation, metrics, baselines and tables.
