# Evaluation plan

## Setting
- **Learning setting: transductive.** Embeddings are learned once on the whole graph, test nodes included, with the target relation removed; only the classifier sees labels. RDF2Vec only gives vectors to nodes in the graph, and KGBench, the R-GCN/RDF2Vec benchmarks and the GeoRDF2Vec paper all use this setup, so our results stay comparable.

## Splits
- **dmg777k: official KGBench splits** (4,394 train / 1,001 validation / 3,001 test). Tune on validation and use test once at the end. This matches the paper's Table 2, and 3,001 test nodes give stable estimates (about ±0.02).
- **BGS+: repeated stratified 5-fold cross-validation, 10 repeats, over training + validation (117 units).** With only 23 validation and 29 test units, one unit moves accuracy by 3–4 points. We also report the official test split, but only for comparison with published results.
- **Random seeds: at least 5 embedding seeds per configuration.** Random walks and word2vec are stochastic, so a single run can't separate a method's effect from run-to-run noise.
- **Robustness check: spatial block cross-validation.** We split by map region so test units lie far from training units. With spatially correlated labels, random splits can reward copying nearby labels rather than generalising (Roberts et al., 2017).

## Metrics
- **Primary: macro F1.** Classes are imbalanced (dmg777k 3.8–47.8%, BGS+ 64/36), and macro F1 penalises ignoring small classes. The paper reports it too.
- **Secondary: accuracy and MCC.** Accuracy is what the paper, KGBench and the published BGS results report; we always show the majority-class baseline next to it (0.478 / 0.637). MCC is in the paper and handles class imbalance well.
- **Diagnostic: per-class F1.** It shows which classes geography helps, especially the rare classes of dmg777k.

## Flooding check (no labels needed)
- **Hold-out of geometry links.** We hide 20% of dmg777k monument geometries, and the stratotypes of all 3,290 BGS+ units that have one, over 5 random folds. This extends the paper's own check (Section 5.4). It tests relation-aware flooding directly, without training embeddings.
- **Measures:**
  - median and 90th-percentile distance error;
  - share of nodes placed in the correct municipality (dmg777k);
  - share of nodes placed at the national centroid.

  Distance shows how far off flooding is, containment matches the paper's own measure, and the centroid share tracks the hub problem we found.

## Compared methods
- **Majority class and kNN on coordinates.** These set the floor and show how much location alone carries.
- **RDF2Vec (uniform walks) and GeoRDF2Vec as published.** These are the paper's baseline and method, with the word2vec bug fixed, since as committed it trains on no data.
- **GeoRDF2Vec** Fixed version of the paper's method.
- **Our additions:** relation-aware flooding, plus adaptive walk length, plus both.These are our contributions, tested separately and together.
