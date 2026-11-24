"""Train one embedding configuration for some seeds and score it on the validation data.

    uv run python -m experiments.run dmg777k rdf2vec_forward --seeds 0 1 2 3 4
    uv run python -m experiments.run bgs rdf2vec_both --seeds 0 1 2 3 4

Each (dataset, configuration, seed) writes experiments/results/<dataset>/<config>_seed<k>.json and
stops early if that file exists, so an interrupted batch resumes where it stopped. The vectors of
the labelled nodes go to data/<dataset>/runs/<config>/seed<k>.npz, so test scores and other
classifiers can be computed later without retraining. Test labels are never read here.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing.load import BGS_DIR, DMG_DIR, ENTITY

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
DATA_DIRS = {"dmg777k": DMG_DIR, "bgs": BGS_DIR}

# direction: "forward" follows edges as stored; "both" adds an inverse edge for every edge between
#   two entities, so a walk ends at the first literal it reaches; "through_literals" adds an inverse
#   edge for every edge, so walks also pass through literals. In both two-way modes a walk never steps
#   straight back to the node it came from.
# start: "all" starts walks from every node, as the authors do; "entities" from entities only.
CONFIGS = {
    "rdf2vec_forward": dict(direction="forward", start="all"),  # E0: RDF2Vec as in the authors' code
    "rdf2vec_both": dict(direction="both", start="entities"),  # E0: the same with E1's walk rules
    "rdf2vec_through_literals": dict(direction="through_literals", start="entities", return_at_dead_end=True),  # E0: walks pass through literals
}
DEFAULTS = dict(walks_per_node=100, depth=8, vector_size=200, window=5, epochs=10)
C_GRID = [0.001, 0.01, 0.1, 1, 10]
CHUNK_STARTS = 20_000  # start nodes per batch of walks, to bound memory


class Graph:
    """Out-edges in CSR form; nodes and relations are integer ids."""

    def __init__(self, triples: pd.DataFrame, direction: str):
        keys, inverse = np.unique(np.concatenate([triples["subject"].to_numpy(), triples["object"].to_numpy()]), return_inverse=True)
        n_triples = len(triples)
        src, dst = inverse[:n_triples], inverse[n_triples:]
        predicates, rel = np.unique(triples["predicate"].to_numpy(), return_inverse=True)
        rel_names = list(predicates)
        is_entity_object = triples["object_kind"].eq(ENTITY).to_numpy()
        if direction == "through_literals":
            is_entity_object = np.ones(n_triples, bool)  # every edge gets an inverse
        if direction in ("both", "through_literals"):
            src, dst = np.concatenate([src, dst[is_entity_object]]), np.concatenate([dst, src[is_entity_object]])
            rel = np.concatenate([rel, rel[is_entity_object] + len(predicates)])
            rel_names += ["inv:" + p for p in predicates]
        elif direction != "forward":
            raise ValueError(direction)
        order = np.argsort(src, kind="stable")
        self.dst, self.rel = dst[order], rel[order]
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=len(keys)))])
        self.keys = keys
        self.rel_names = rel_names
        self.is_entity = np.zeros(len(keys), bool)
        self.is_entity[inverse[:n_triples]] = True  # subjects are always entities
        self.is_entity[inverse[n_triples:][triples["object_kind"].eq(ENTITY).to_numpy()]] = True

    def degree(self, nodes: np.ndarray) -> np.ndarray:
        return self.indptr[nodes + 1] - self.indptr[nodes]


def random_walks(graph: Graph, starts: np.ndarray, walks_per_node: int, depth: int, no_backtrack: bool, rng: np.random.Generator,
                 return_at_dead_end: bool = False):
    """Uniform random walks for a batch of start nodes. Returns node ids (walks, depth+1) and relation ids (walks, depth); -1 after a walk ends.

    With no_backtrack, a walk never steps straight back to the node it came from. If that is the only way on,
    it stops, or with return_at_dead_end steps back after all.
    """
    cur = np.repeat(starts, walks_per_node)
    prev = np.full_like(cur, -1)
    alive = np.ones(len(cur), bool)
    nodes, rels = [cur.copy()], []
    for _ in range(depth):
        deg = graph.degree(cur)
        alive &= deg > 0
        edge = np.zeros(len(cur), np.int64)
        edge[alive] = graph.indptr[cur[alive]] + (rng.random(alive.sum()) * deg[alive]).astype(np.int64)
        if no_backtrack:
            # Redraw walks that chose the node they came from; after 10 tries, assume it is the only way on.
            for _ in range(10):
                back = alive & (graph.dst[edge] == prev)
                if not back.any():
                    break
                edge[back] = graph.indptr[cur[back]] + (rng.random(back.sum()) * deg[back]).astype(np.int64)
            if not return_at_dead_end:
                alive &= graph.dst[edge] != prev
        nxt = np.where(alive, graph.dst[edge], -1)
        rels.append(np.where(alive, graph.rel[edge], -1))
        nodes.append(nxt)
        prev, cur = np.where(alive, cur, prev), np.where(alive, nxt, cur)
    return np.stack(nodes, 1), np.stack(rels, 1)


def write_walks(nodes: np.ndarray, rels: np.ndarray, handle) -> int:
    """Append walks as lines of tokens (n<id> r<id> n<id> ...); walks without a single step are skipped. Returns tokens written."""
    steps = (rels >= 0).sum(1)
    tokens = 0
    for i in np.flatnonzero(steps > 0):
        k = steps[i]
        line = [f"n{nodes[i, 0]}"]
        for j in range(k):
            line.append(f"r{rels[i, j]}")
            line.append(f"n{nodes[i, j + 1]}")
        tokens += len(line)
        handle.write(" ".join(line) + "\n")
    return tokens


def evaluate(dataset: str, X: np.ndarray, y: np.ndarray, split: np.ndarray, seed: int) -> dict:
    """Validation scores with logistic regression. dmg777k: train -> valid, C tuned on valid.
    BGS+: repeated stratified 5-fold CV (10 repeats) over train + valid, C tuned by inner 3-fold CV."""
    model = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))  # noqa: E731
    scores = lambda t, p: {"macro_f1": f1_score(t, p, average="macro"), "accuracy": accuracy_score(t, p), "mcc": matthews_corrcoef(t, p)}  # noqa: E731
    if dataset == "dmg777k":
        tr, va = split == "train", split == "valid"
        best = None
        for C in C_GRID:
            s = scores(y[va], model(C).fit(X[tr], y[tr]).predict(X[va]))
            if best is None or s["macro_f1"] > best["macro_f1"]:
                best = {**s, "C": C}
        return {"protocol": "train->valid", **best}
    use = np.isin(split, ["train", "valid"])
    X, y = X[use], y[use]
    folds = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed).split(X, y):
        search = GridSearchCV(model(1.0), {"logisticregression__C": C_GRID}, cv=3, scoring="f1_macro").fit(X[tr], y[tr])
        folds.append(scores(y[te], search.predict(X[te])))
    folds = pd.DataFrame(folds)
    out = {"protocol": "5-fold x10 CV on train+valid"}
    for m in folds:
        out[m], out[m + "_std"] = float(folds[m].mean()), float(folds[m].std())
    return out


def run(dataset: str, config: str, seed: int, smoke: bool) -> None:
    params = {**DEFAULTS, **CONFIGS[config]}
    if smoke:
        params.update(walks_per_node=2, epochs=1)
    name = f"{config}_seed{seed}" + ("_smoke" if smoke else "")
    result_path = RESULTS_DIR / dataset / f"{name}.json"
    if result_path.exists():
        print(f"{result_path.relative_to(REPO_ROOT)} exists, skipping")
        return
    data_dir = DATA_DIRS[dataset]
    run_dir = data_dir / "runs" / config
    run_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    t = time.time()
    processed = data_dir / "processed"
    graph = Graph(pd.read_parquet(processed / "triples.parquet"), params["direction"])
    starts = np.arange(len(graph.keys)) if params["start"] == "all" else np.flatnonzero(graph.is_entity)
    timings["graph_s"] = time.time() - t

    t = time.time()
    rng = np.random.default_rng(seed)
    walk_file = run_dir / f"{name}.walks.txt"
    tokens = 0
    with walk_file.open("w") as handle:
        for i in range(0, len(starts), CHUNK_STARTS):
            nodes, rels = random_walks(graph, starts[i : i + CHUNK_STARTS], params["walks_per_node"], params["depth"], params["direction"] != "forward", rng,
                                         params.get("return_at_dead_end", False))
            tokens += write_walks(nodes, rels, handle)
    timings["walks_s"] = time.time() - t

    t = time.time()
    model = Word2Vec(corpus_file=str(walk_file), vector_size=params["vector_size"], window=params["window"], min_count=0,
                     workers=os.cpu_count(), epochs=params["epochs"], seed=seed)
    walk_file.unlink()
    timings["word2vec_s"] = time.time() - t

    labels = pd.read_csv(processed / "labels.csv")
    node_id = pd.Series(np.arange(len(graph.keys)), index=graph.keys)
    ids = node_id.reindex(labels["node"]).to_numpy()
    tokens_of = [f"n{int(i)}" if not np.isnan(i) else None for i in ids]
    missing = sum(tok is None or tok not in model.wv.key_to_index for tok in tokens_of)
    if missing:
        raise RuntimeError(f"{missing} labelled nodes have no vector")
    X = np.stack([model.wv[tok] for tok in tokens_of])
    np.savez_compressed(run_dir / f"{name}.npz", node=labels["node"].to_numpy(), vectors=X)

    t = time.time()
    metrics = evaluate(dataset, X, labels["class"].astype(str).to_numpy(), labels["split"].to_numpy(), seed)
    timings["classifier_s"] = time.time() - t

    result = {"dataset": dataset, "config": config, "seed": seed, "params": params, "smoke": smoke,
              "machine": platform.node(), "platform": platform.platform(), "cpus": os.cpu_count(),
              "start_nodes": int(len(starts)), "tokens": int(tokens), "timings": timings, "validation": metrics}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"{dataset} {name}: macro F1 {metrics['macro_f1']:.3f}, accuracy {metrics['accuracy']:.3f}, "
          f"word2vec {timings['word2vec_s'] / 60:.1f} min, {tokens / 1e6:.0f}M tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(DATA_DIRS))
    parser.add_argument("config", choices=sorted(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true", help="2 walks per node and 1 epoch, to test the pipeline end to end")
    args = parser.parse_args()
    for seed in args.seeds:
        run(args.dataset, args.config, seed, args.smoke)


if __name__ == "__main__":
    main()
