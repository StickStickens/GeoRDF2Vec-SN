"""Score saved embeddings again without retraining: baselines, spatial block cross-validation, test set.

    uv run python -m experiments.evaluate dmg777k bgs            # baselines + random vs spatial block CV
    uv run python -m experiments.evaluate dmg777k bgs --test geo_best rdf2vec_through_literals ...

Cross-validation runs over the train + valid nodes only, twice per embedding with the same classifier:
folds drawn at random (stratified), and folds made of spatial blocks, so that no test node has a
training node from its own block. The blocks are k-means clusters of the nodes' locations (the
relation-aware flooding, which puts every labelled node somewhere). If an embedding scores much worse
on spatial blocks, part of its score came from nearby nodes sharing a class.

The classifier is the one run.py uses: standardised vectors and logistic regression. C is tuned by an
inner 3-fold CV of the same kind (grouped by block for the block CV) on BGS+. On dmg777k that would
take hours, so C is the value run.py chose on the validation split for that embedding.

Baselines, on the same folds: the majority class, and kNN on the coordinates alone (k tuned the same
way as C). They show how much class information location carries by itself.

--test reads the test labels, once, for the configurations chosen on validation: fit on train + valid
(C as above, on BGS+ tuned by 5-fold CV on train + valid), score on test, per seed.

Writes experiments/results/<dataset>/spatial_cv.csv (one row per embedding, protocol and repeat) and
final_test.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.run import C_GRID, CONFIGS, DATA_DIRS, DEFAULTS, RESULTS_DIR, Graph, flood_graph, load_triples

FOLDS = 5
REPEATS = 5  # block layouts (k-means seeds) and random fold draws
BLOCKS = {"dmg777k": 25, "bgs": 15}  # about 5 and 3 blocks per fold
K_GRID = [1, 3, 5, 10, 20, 50]
SEEDS = range(5)
# Embeddings compared by default: E0, E1 (b), the E2-E5 factors E8 combines, and E8.
DEFAULT_CONFIGS = ["rdf2vec_forward", "rdf2vec_through_literals", "geo_published", "geo_through_literals",
                   "geo_relation_aware", "geo_hubs10", "geo_nolang", "geo_best", "geo_best_hubs10"]


def scores(y_true, y_pred) -> dict:
    return {"macro_f1": f1_score(y_true, y_pred, average="macro"), "accuracy": accuracy_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred)}


def logistic(C: float = 1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def labelled_locations(dataset: str, nodes: pd.Series) -> np.ndarray:
    """(x, y) in km of each labelled node: its location after relation-aware flooding (cached), on a plane
    tangent at the nodes' mean position. Good enough to cut a country into blocks."""
    processed = DATA_DIRS[dataset] / "processed"
    params = {**DEFAULTS, **CONFIGS["geo_relation_aware"]}
    graph = Graph.from_triples(load_triples(processed, params, {}), "forward")
    flooded = flood_graph(graph, params, processed, DATA_DIRS[dataset] / "runs" / "_flood", {})
    ids = pd.Series(np.arange(len(graph.keys)), index=graph.keys).reindex(nodes).to_numpy()
    if np.isnan(ids.astype(float)).any() or np.isnan(flooded.lon[ids]).any():
        raise RuntimeError("a labelled node has no location")
    lon, lat = flooded.lon[ids], flooded.lat[ids]
    return np.column_stack([np.radians(lon - lon.mean()) * np.cos(np.radians(lat.mean())), np.radians(lat - lat.mean())]) * 6371.0


def folds(protocol: str, y: np.ndarray, xy: np.ndarray, repeat: int, dataset: str):
    """(train, test, groups of the train part) per fold."""
    if protocol == "random":
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=repeat).split(xy, y):
            yield tr, te, None
        return
    blocks = KMeans(BLOCKS[dataset], n_init=10, random_state=repeat).fit_predict(xy)
    for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=repeat).split(xy, y, blocks):
        yield tr, te, blocks[tr]


def fit_predict(model, grid: dict | None, X_tr, y_tr, groups, X_te):
    """Fit, tuning over grid by an inner 3-fold CV (grouped by block when groups is given)."""
    if grid is None:
        return model.fit(X_tr, y_tr).predict(X_te)
    inner = StratifiedGroupKFold(3) if groups is not None else StratifiedKFold(3)
    search = GridSearchCV(model, grid, cv=inner, scoring="f1_macro").fit(X_tr, y_tr, groups=groups)
    return search.predict(X_te)


def cross_validate(name: str, make, grid, X, y, xy, dataset: str) -> list[dict]:
    rows = []
    for protocol in ["random", "spatial_blocks"]:
        for repeat in range(REPEATS):
            per_fold = [scores(y[te], fit_predict(make(), grid, X[tr], y[tr], g, X[te])) for tr, te, g in folds(protocol, y, xy, repeat, dataset)]
            rows.append({"embedding": name, "protocol": protocol, "repeat": repeat, **pd.DataFrame(per_fold).mean().to_dict()})
    return rows


def load_embedding(dataset: str, config: str, seed: int, nodes: pd.Series) -> tuple[np.ndarray, dict] | None:
    path = DATA_DIRS[dataset] / "runs" / config / f"{config}_seed{seed}.npz"
    result = RESULTS_DIR / dataset / f"{config}_seed{seed}.json"
    if not path.exists() or not result.exists():
        return None
    z = np.load(path, allow_pickle=True)
    if not np.array_equal(z["node"], nodes.to_numpy()):
        raise RuntimeError(f"{path}: nodes differ from labels.csv")
    return z["vectors"], json.loads(result.read_text())["validation"]


def spatial_cv(dataset: str, configs: list[str], jobs: int) -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIRS[dataset] / "processed" / "labels.csv")
    use = labels["split"].isin(["train", "valid"]).to_numpy()
    y = labels["class"].astype(str).to_numpy()[use]
    xy = labelled_locations(dataset, labels["node"])[use]
    print(f"{dataset}: {use.sum():,} train + valid nodes, {BLOCKS[dataset]} blocks, {FOLDS} folds x {REPEATS} repeats", flush=True)

    tasks = [delayed(cross_validate)("majority class", lambda: DummyClassifier(strategy="most_frequent"), None, xy, y, xy, dataset),
             delayed(cross_validate)("kNN on coordinates", KNeighborsClassifier, {"n_neighbors": K_GRID}, xy, y, xy, dataset)]
    for config in configs:
        for seed in SEEDS:
            loaded = load_embedding(dataset, config, seed, labels["node"])
            if loaded is None:
                print(f"  {config} seed {seed}: no embedding, skipped")
                continue
            X, validation = loaded
            if dataset == "dmg777k":  # C as chosen on the validation split
                make, grid = (lambda C=validation["C"]: logistic(C)), None
            else:
                make, grid = logistic, {"logisticregression__C": C_GRID}
            tasks.append(delayed(cross_validate)(f"{config}_seed{seed}", make, grid, X[use], y, xy, dataset))
    rows = [r for part in Parallel(n_jobs=jobs, verbose=5)(tasks) for r in part]
    out = pd.DataFrame(rows)
    out.insert(0, "config", out["embedding"].str.replace(r"_seed\d+$", "", regex=True))
    out.to_csv(RESULTS_DIR / dataset / "spatial_cv.csv", index=False)

    summary = out.groupby(["config", "protocol"], sort=False)["macro_f1"].agg(["mean", "std"]).unstack("protocol")
    summary[("drop", "")] = summary[("mean", "random")] - summary[("mean", "spatial_blocks")]
    print(summary.round(3).to_string(), flush=True)
    return out


def test(dataset: str, configs: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIRS[dataset] / "processed" / "labels.csv")
    y = labels["class"].astype(str).to_numpy()
    fit, te = labels["split"].isin(["train", "valid"]).to_numpy(), (labels["split"] == "test").to_numpy()
    xy = labelled_locations(dataset, labels["node"])
    majority = DummyClassifier(strategy="most_frequent").fit(xy[fit], y[fit])
    knn = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": K_GRID}, cv=StratifiedKFold(5), scoring="f1_macro").fit(xy[fit], y[fit])
    rows = [{"config": "majority class", "seed": None, **scores(y[te], majority.predict(xy[te]))},
            {"config": "kNN on coordinates", "seed": None, "k": knn.best_params_["n_neighbors"], **scores(y[te], knn.predict(xy[te]))}]
    for config in configs:
        for seed in SEEDS:
            loaded = load_embedding(dataset, config, seed, labels["node"])
            if loaded is None:
                raise RuntimeError(f"{dataset} {config} seed {seed}: no embedding")
            X, validation = loaded
            if dataset == "dmg777k":
                model = logistic(validation["C"]).fit(X[fit], y[fit])
            else:
                model = GridSearchCV(logistic(), {"logisticregression__C": C_GRID}, cv=StratifiedKFold(5), scoring="f1_macro").fit(X[fit], y[fit])
            rows.append({"config": config, "seed": seed, **scores(y[te], model.predict(X[te]))})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / dataset / "final_test.csv", index=False)
    print(f"{dataset}: {te.sum()} test nodes")
    print(out.groupby("config", sort=False)[["macro_f1", "accuracy", "mcc"]].agg(["mean", "std"]).round(3).to_string())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", choices=sorted(DATA_DIRS))
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="embeddings for the cross-validation")
    parser.add_argument("--test", nargs="+", metavar="CONFIG", help="score these configurations on the test set instead")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    for dataset in args.datasets:
        if args.test:
            test(dataset, args.test)
        else:
            spatial_cv(dataset, args.configs, args.jobs)


if __name__ == "__main__":
    main()
