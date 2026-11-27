# Experiments (Milestone 4)

We re-implemented GeoRDF2Vec, fixed the bugs we found in the authors' code, and tested nine ideas for handling hubs, locations and walks (E0–E8) on dmg777k and BGS+. Every configuration was trained with 5 seeds. In total that is 22 configurations × 5 seeds × 2 datasets = 220 embeddings.

| File | Contents |
|---|---|
| [evaluation_plan.md](evaluation_plan.md) | Learning setting, splits, metrics and compared methods, as planned |
| [run.py](run.py) | Builds the graph, floods geometries, generates walks, trains word2vec and scores on validation; `--list` shows the configurations |
| [geo.py](geo.py) | Relation-aware flooding, flooding, spatial edge weights |
| [flood_check.py](flood_check.py) | Label-free check of flooding: hide known locations, flood, measure the error (`--sweep` for the threshold sensitivity) |
| [evaluate.py](evaluate.py) | Scores the saved vectors again without retraining: baselines, random vs spatial block cross-validation, test set (`--test`) |
| `results/<dataset>/` | One JSON per run (parameters, machine, walk statistics, timings, validation scores); `flood_check.csv`, `flood_sweep.csv`, `spatial_cv.csv`, `final_test.csv` |

Input: `data/<dataset>/processed/`, written by [preprocessing](../preprocessing/README.md). Vectors of the labelled nodes: `data/<dataset>/runs/<config>/<config>_seed<k>.npz`.

## Summary

- **Fixing the method matters far more than any geographic addition.** Our best configuration beats GeoRDF2Vec as published by 0.13 macro F1 on the dmg777k test set (0.691 vs 0.564) and by 0.20 on the BGS+ test set (0.758 vs 0.561). Most of this comes from letting walks pass through literals in both directions (E1). Our other fixes, kept with forward walks, do not help on their own (0.610 → 0.597 on dmg777k, 0.592 → 0.529 on BGS+, validation).
- **Geography itself adds little.** With the same walks, adding the authors' flooding and spatial weights changes macro F1 by −0.005 on dmg777k and +0.012 on BGS+ (validation).
  - On dmg777k, our best combination, `geo_best_hubs10`, beats RDF2Vec without geography on the official split: +0.028 on validation, +0.029 on test. That gain does not appear in cross-validation over train + valid (+0.001).
  - On BGS+, every top configuration is within noise of RDF2Vec, and on its 29 test nodes RDF2Vec scores highest.
- **Location carries little class information on dmg777k. On BGS+ it carries a lot, but only locally.**
  - On BGS+, kNN on coordinates alone scores 0.789 under random folds, more than any embedding. On unseen map regions it falls to 0.536.
  - The embeddings lose much less on unseen regions: 0.02–0.03 on dmg777k and 0.10–0.14 on BGS+.
- **Relation-aware flooding fixes the flooding** (BGS+ median error 238 → 32 km, no nodes left at the national centroid), but it hardly changes classification (+0.016 and −0.001).
- **Hubs carry class information.** Down-weighting high-degree nodes (E4) costs 0.11–0.19 macro F1, and removing 50 hubs costs 0.08 on dmg777k. Removing only the top 10 is neutral on its own and helps in combination.
- **Adaptive walk length (E7) hurts.** Stopping walks after they travel *d* km costs 0.08–0.20. On dmg777k, a control with uniformly short walks loses only 0.014. So the loss comes from the long jumps the distance rules cut, which link a monument to distant monuments of the same photographer or collection.

## Setup

**Graph.** Our preprocessed triples, with the target relation removed. A node is an entity or a literal. Embeddings never see labels (transductive setting, see [evaluation_plan.md](evaluation_plan.md)).

**Walks.** A vectorised walker in [run.py](run.py), 100 walks of depth 8 per start node.
- *Forward* walks follow edges as stored and start at every node, as in the authors' code.
- *Both-way* walks start at entities, add inverse edges and never step straight back (up to 10 redraws). There are two variants:
  - (a) inverse edges only between entities, so a walk ends at the first literal;
  - (b) inverse edges for every edge, so walks pass through literals, and a walk steps back only at a dead end.
- Walks of one token are dropped: they give word2vec no training pairs.

**Geography.** Geometries are spread to nodes without one by flooding (multi-source breadth-first search, as in the authors' `neighborhood_flood`). A node's location is the centroid of the geometries it inherits. A walk steps along an edge with probability proportional to exp(−scaled distance), min-max scaled over the edges available at the current node.

**word2vec.** gensim CBOW, 200 dimensions, window 5, 10 epochs, `min_count` 0, trained from a walk file (`corpus_file`).

**Classifier and protocols.** Standardised vectors, logistic regression, C ∈ {0.001, 0.01, 0.1, 1, 10}. Primary metric: macro F1; accuracy and MCC are in the result files.
- *dmg777k:* official KGBench splits. C is chosen on validation (5,394 train → 1,001 valid); the test set (2,001) is scored once at the end.
- *BGS+:* 5-fold cross-validation repeated 10 times over train + valid (117 units), with C tuned by an inner 3-fold CV. The official test split (29 units) is scored once at the end.
- *Seeds:* 5 per configuration. Tables show the mean ± standard deviation over seeds.

Logistic regression was chosen over XGBoost after a comparison on one dmg777k embedding. It was as accurate and 20× faster to tune:

| Classifier | Macro F1 | Accuracy | Tuning time |
|---|---|---|---|
| Majority class | — | 0.478 | — |
| Logistic regression, 5 values of C | **0.702** | **0.747** | 29 s |
| XGBoost, default settings | 0.636 | 0.712 | 9 s |
| XGBoost, 20-configuration random search | 0.697 | 0.735 | 553 s |

## What we changed in the authors' code

Found in the EDA and in a reading of `original/geo_rdf2vec.py`, which we kept unchanged.

1. **word2vec trains on nothing.** `corpus` is an iterator that `build_vocab` uses up, so `train` sees 0 words. We write walks to a file and train from it, which also keeps memory bounded.
2. **Longitude and latitude are swapped** when distances are computed (geopy expects (lat, lon)). For example, Amsterdam–Utrecht comes out as 39.4 km instead of 34.6 km.
3. **Geometry detection is case-sensitive** and misses `Point(...)`. Our preprocessing writes WKT keywords in capitals.
4. **The min-max edge weights are wrong and slow.** Each edge is normalised once from each end, and the last write wins. The loop scans the whole edge table for every node. We normalise at each node over the edges a walk can take next (`minmax_weights` in [geo.py](geo.py)). `geo_published` keeps the authors' behaviour (`authors_minmax_weights`), which we checked against the EDA's re-implementation on all 916k BGS+ edges.

`geo_published` is GeoRDF2Vec as published: the authors' graph, flooding, weights and forward walks, with only bug 1 fixed.

## Experiments

Each experiment changes one factor against a fixed configuration. From E2 on, that is `geo_through_literals`: walks of type (b), the authors' flooding, our min-max weights.

| # | Factor | Configurations |
|---|---|---|
| E0 | Baselines: majority class, kNN on coordinates; RDF2Vec (uniform walks) with each walk type; GeoRDF2Vec as published | `rdf2vec_forward`, `rdf2vec_both`, `rdf2vec_through_literals`, `geo_published` |
| E1 | Walk direction, with geography: forward, (a) both ways ending at literals, (b) both ways through literals | `geo_forward`, `geo_both`, `geo_through_literals` |
| E2 | Relation-aware flooding: only relations whose objects lie close together pass on locations | `geo_relation_aware` |
| E3 | Remove the top k hubs, k ∈ {10, 50} | `geo_hubs10`, `geo_hubs50` |
| E4 | Degree down-weighting: edge weight × degree(target)^−α, α ∈ {0.5, 1} | `geo_degree05`, `geo_degree1` |
| E5 | Strip language tags, so literals with identical text become one node | `geo_nolang` |
| E6 | 50 or 200 walks for nodes with their own geometry, instead of 100 | `geo_walks50`, `geo_walks200` |
| E7 | Adaptive walk length: four stopping rules; control: depth 4 | `geo_len_cap`, `geo_len_soft`, `geo_len_budget`, `geo_len_unreliable`, `geo_depth4` |
| E8 | Combination of the factors that did not hurt on either dataset (E1 (b), E2, E5), with and without E3's 10-hub removal | `geo_best`, `geo_best_hubs10` |

**Details.**
- **E2, relation-aware flooding.** A relation passes on locations if the located subjects sharing one of its objects lie within 0.4 × the dataset's baseline spread of their group centre: 30 km on dmg777k, 99 km on BGS+. The baseline spread is the median distance of located entities from their mean position. Where too few subjects are located to measure this, a relation passes if its objects are shared by at most 20 subjects on average. Relations that carry geometries always pass. This keeps 40 of 60 relations on dmg777k and 70 of 103 on BGS+. The 0.4 was chosen by hand, without labels; the sensitivity check below shows the result is stable around it.
- **E3, hub removal.** We remove the k highest-degree nodes, skipping labelled nodes and any hub whose removal would leave a labelled node without edges. The only node skipped is one labelled dmg777k monument that is itself among the top 50 hubs.
- **E5, language tags.** Translations are not merged. Both datasets are nearly monolingual, so the merges join tagged and untagged copies of the same text, such as `"Aalsmeer"@nl` and `"Aalsmeer"`. That is 266 groups on dmg777k and 20 on BGS+, affecting 753 and 4,034 triples.
- **E7, stopping rules.** *d* is the same 0.4 × baseline spread (30 / 99 km).
  1. *Cap:* stop once the walk is more than *d* from its start.
  2. *Soft:* stop at each step with probability 1 − exp(−distance / *d*).
  3. *Unreliable:* stop on a node that inherits more than half of all geometries, i.e. is flooded to near the national centroid.
  4. *Budget:* stop once the summed step lengths exceed *d*.

## Results

### Validation

Macro F1, mean ± std over 5 seeds. dmg777k: train → valid. BGS+: repeated 5-fold CV over train + valid. Δ is the difference to `geo_through_literals`, from E2 on.

| # | Configuration | dmg777k | Δ | BGS+ | Δ |
|---|---|---|---|---|---|
| E0 | rdf2vec_forward | 0.627 ± 0.010 | | 0.548 ± 0.023 | |
| E0 | rdf2vec_both | 0.654 ± 0.010 | | 0.718 ± 0.007 | |
| E0 | rdf2vec_through_literals | 0.677 ± 0.012 | | 0.761 ± 0.044 | |
| E0 | geo_published | 0.610 ± 0.013 | | 0.592 ± 0.036 | |
| E1 | geo_forward | 0.597 ± 0.009 | | 0.529 ± 0.040 | |
| E1 | geo_both | 0.616 ± 0.018 | | 0.718 ± 0.042 | |
| E1 | geo_through_literals | 0.672 ± 0.018 | — | 0.773 ± 0.012 | — |
| E2 | geo_relation_aware | 0.688 ± 0.009 | +0.016 | 0.772 ± 0.044 | −0.001 |
| E3 | geo_hubs10 | 0.686 ± 0.021 | +0.014 | 0.755 ± 0.037 | −0.018 |
| E3 | geo_hubs50 | 0.590 ± 0.018 | −0.082 | 0.756 ± 0.055 | −0.017 |
| E4 | geo_degree05 | 0.542 ± 0.014 | −0.130 | 0.660 ± 0.018 | −0.113 |
| E4 | geo_degree1 | 0.486 ± 0.010 | −0.186 | 0.646 ± 0.019 | −0.127 |
| E5 | geo_nolang | 0.674 ± 0.008 | +0.002 | 0.777 ± 0.024 | +0.004 |
| E6 | geo_walks50 | 0.679 ± 0.009 | +0.007 | 0.744 ± 0.022 | −0.029 |
| E6 | geo_walks200 | 0.664 ± 0.015 | −0.008 | 0.753 ± 0.046 | −0.020 |
| E7 | geo_len_cap | 0.489 ± 0.020 | −0.183 | 0.677 ± 0.049 | −0.096 |
| E7 | geo_len_soft | 0.519 ± 0.012 | −0.153 | 0.690 ± 0.042 | −0.083 |
| E7 | geo_len_budget | 0.470 ± 0.009 | −0.202 | 0.694 ± 0.020 | −0.079 |
| E7 | geo_len_unreliable | 0.682 ± 0.009 | +0.010 | 0.670 ± 0.024 | −0.103 |
| E7 | geo_depth4 (control) | 0.658 ± 0.009 | −0.014 | 0.692 ± 0.037 | −0.081 |
| E8 | geo_best | 0.681 ± 0.008 | +0.009 | 0.756 ± 0.061 | −0.017 |
| E8 | **geo_best_hubs10** | **0.705 ± 0.004** | +0.033 | **0.795 ± 0.049** | +0.022 |

On BGS+, the seed-to-seed standard deviation of most configurations is 0.02–0.06, so differences below about 0.05 there are not reliable.

### Test

For the test set, one configuration per method was chosen on validation: RDF2Vec, GeoRDF2Vec as published, GeoRDF2Vec with our fixes, and our best combination. Each was scored once, fitting the classifier on train + valid. The baselines do not depend on a seed.

| | dmg777k (2,001 nodes) | BGS+ (29 nodes) |
|---|---|---|
| Majority class | 0.129 | 0.396 |
| kNN on coordinates | 0.280 | 0.780 |
| rdf2vec_through_literals | 0.662 ± 0.024 | **0.824 ± 0.021** |
| geo_published | 0.564 ± 0.010 | 0.561 ± 0.036 |
| geo_through_literals | 0.654 ± 0.021 | 0.742 ± 0.058 |
| geo_best_hubs10 | **0.691 ± 0.008** | 0.758 ± 0.062 |

On dmg777k the validation ranking holds. On BGS+ one test node is 3.4 accuracy points, and the gap between RDF2Vec and `geo_best_hubs10` is about two nodes. The BGS+ cross-validation above, over 117 nodes, is the more reliable comparison.

### Spatial block cross-validation

With random folds, neighbouring nodes end up in both train and test. Neighbours tend to share a class, so a model that knows where a node is gets credit for that alone. [evaluate.py](evaluate.py) scores each saved embedding twice over train + valid with the same classifier:
- 5 random stratified folds;
- 5 folds of spatial blocks: k-means clusters of the locations from relation-aware flooding, 25 on dmg777k and 15 on BGS+, so no test node shares a block with a training node.

Each protocol is repeated 5 times. The table shows macro F1, mean over 5 seeds × 5 repeats.

| | dmg777k random | dmg777k blocks | BGS+ random | BGS+ blocks |
|---|---|---|---|---|
| Majority class | 0.129 | 0.129 | 0.387 | 0.383 |
| kNN on coordinates | 0.448 | 0.217 | **0.789** | 0.536 |
| rdf2vec_forward | 0.725 | 0.703 | 0.554 | 0.478 |
| rdf2vec_through_literals | 0.775 | 0.753 | 0.762 | 0.662 |
| geo_published | 0.707 | 0.680 | 0.604 | 0.556 |
| geo_through_literals | 0.756 | 0.730 | 0.775 | 0.675 |
| geo_relation_aware | 0.779 | 0.758 | 0.773 | 0.634 |
| geo_hubs10 | **0.785** | **0.761** | 0.761 | 0.644 |
| geo_nolang | 0.766 | 0.742 | 0.781 | 0.681 |
| geo_best | 0.767 | 0.750 | 0.760 | 0.626 |
| geo_best_hubs10 | 0.776 | 0.755 | **0.795** | **0.684** |

- On dmg777k, kNN on coordinates falls almost to the majority class on unseen areas, while every embedding loses only 0.02–0.03.
- On BGS+, kNN loses 0.25, and the embeddings lose 0.05–0.14. `geo_relation_aware` and `geo_best` lose the most (0.13–0.14).
- dmg777k scores are higher here than on the official split (0.775 against 0.677 for `rdf2vec_through_literals`): the folds train on 80% of train + valid, and the official validation split seems harder than a random draw. Compare configurations within one protocol only.
- On dmg777k, C is the value chosen on validation for each embedding, because nested tuning would take hours. On BGS+ it is tuned by an inner 3-fold CV, grouped by block in the block protocol.

### Flooding check (no labels)

Known locations are hidden in 5 folds, flooding is run again, and the error is measured where flooding puts them:
- on dmg777k, the `hasGeometry` links of the 10,303 entities with their own geometry;
- on BGS+, the `hasStratotype` links of the 3,290 units with a stratotype. The truth is the mean location of their stratotypes.

| Dataset | Flooding | Median error | 90th percentile | At the national centroid |
|---|---|---|---|---|
| dmg777k | all relations (authors) | 2.3 km | 7.4 km | 0% |
| dmg777k | relation-aware | 2.2 km | 6.5 km | 0% |
| BGS+ | all relations (authors) | 238 km | 480 km | 98% |
| BGS+ | relation-aware | 32 km | 224 km | 0% |

On dmg777k, hidden monuments still get their location from their address and city, so flooding was already accurate. On BGS+, the authors' flooding sends almost every hidden unit to the national centroid, through relations such as age and colour that are shared across the whole country.

**Sensitivity to the 0.4 threshold** (`flood_check.py --sweep`). Each row is one distinct set of passing relations:

| Dataset | Spread fraction | Median error | 90th percentile | Nodes with a location |
|---|---|---|---|---|
| dmg777k | 0.05–0.8 (incl. 0.4) | 2.2 km | 6.5 km | 81.5–82.1% |
| dmg777k | 1.0 | 2.3 km | 7.3 km | 82.5% |
| BGS+ | 0.05–0.34 | 30 km | 234 km | 80.9% |
| BGS+ | 0.35–0.45 (incl. 0.4) | 32 km | 224 km | 98.0% |
| BGS+ | 0.5–0.8 | 99 km | 353 km | 98.1% |
| BGS+ | 1.0 | 105 km | 371 km | 98.2% |

- On BGS+, any fraction from 0.35 to 0.45 gives the same result.
- Below 0.34, `hasLexicon`, which links the map to the rock units, stops passing, and 17% of nodes get no location.
- From 0.5, colours, ages and rock class pass, and the error triples.
- All labelled nodes get a location at every fraction.

## Conclusions by experiment

- **E0/E1, walk direction, is the largest effect.** Forward walks end after about one step, because most edges lead to literals. Walks of type (b), which pass through literals both ways, beat forward walks by 0.05–0.24 and type (a) by 0.02–0.06, with or without geography.
- **E2, relation-aware flooding,** makes flooding accurate on BGS+ but barely changes classification. Location is not where the class signal is on dmg777k, and on BGS+ what it adds is mostly local, as the spatial blocks show.
- **E3/E4, hubs.** Hubs such as photographers and collections seem to connect nodes of the same class, so weakening every hub (E4) or removing many of them (E3, k = 50 on dmg777k) loses that signal. Removing only the 10 largest is neutral alone (+0.014, −0.018) and gives the best combination with E2 and E5. Those 10 are class nodes such as `Geometry` and `Afbeelding` (image), and the most common techniques, collections and photographers.
- **E5, language tags,** and **E6, walks per located node,** make no measurable difference.
- **E7, adaptive walk length.** The three distance rules hurt on both datasets.
  - On dmg777k, the unreliable-node rule cuts walks to the same length (4.4 steps) as the distance rules and loses nothing, and the depth-4 control loses only 0.014. So the distance rules hurt by cutting specific long jumps, not by making walks shorter.
  - On BGS+, every rule costs about as much as shortening the walks to depth 4.
- **E8.** `geo_best_hubs10` is the best configuration on validation on both datasets, and the best on the dmg777k test set. Its margin over RDF2Vec without geography holds on the official dmg777k split, but not in cross-validation over train + valid, and not on BGS+.

## Not done

- **Correct-municipality share in the flooding check.** It is in [evaluation_plan.md](evaluation_plan.md), but we only measured distance error.
- **Relation-aware flooding together with adaptive walk length.** The plan listed this combination, but E7 hurt on both datasets, so E8 left it out.
- **Per-class F1.** It is planned as a diagnostic, but not reported. Per-run predictions can be recomputed from the saved vectors.
- **XGBoost.** The comparison was not repeated on the final configurations, and the paper's own classifier setup (XGBoost, 60 rounds of Bayesian tuning) was not reproduced. All reported scores use logistic regression.

## Reproducing

```
uv run python -m experiments.run <dataset> <config> [<config> ...] [--seeds 0 1 2 3 4]   # dataset: dmg777k or bgs
uv run python -m experiments.flood_check dmg777k bgs [--sweep]
uv run python -m experiments.evaluate dmg777k bgs                                         # spatial_cv.csv
uv run python -m experiments.evaluate dmg777k bgs --test rdf2vec_through_literals geo_published geo_through_literals geo_best_hubs10
```

A run skips itself if its result file exists, so an interrupted batch resumes where it stopped. Flooding is cached in `data/<dataset>/runs/_flood/`.

**Compute.** One run takes 10–26 minutes, almost all of it word2vec, on a Ryzen 5 3600X (12 threads) or a MacBook Air M1. The 220 runs took about 70 hours of compute, split across the two machines; each result file records the machine it ran on. Spatial cross-validation and test scoring take a few minutes.
