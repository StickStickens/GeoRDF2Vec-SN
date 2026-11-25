"""Train one embedding configuration for some seeds and score it on the validation data.

    uv run python -m experiments.run dmg777k geo_through_literals --seeds 0 1 2 3 4
    uv run python -m experiments.run --list

Each (dataset, configuration, seed) writes experiments/results/<dataset>/<config>_seed<k>.json and
stops early if that file exists, so an interrupted batch resumes where it stopped. The vectors of
the labelled nodes go to data/<dataset>/runs/<config>/seed<k>.npz, so test scores and other
classifiers can be computed later without retraining. Test labels are never read here.

Flooding does not depend on the seed, so its result is cached in data/<dataset>/runs/_flood/.
"""

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eda import spatial as sp
from eda.graph import build_graph
from experiments import geo
from preprocessing.geometry import geodesic_km
from preprocessing.load import BGS_DIR, DMG_DIR, ENTITY

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
DATA_DIRS = {"dmg777k": DMG_DIR, "bgs": BGS_DIR}

# Parameters of a configuration; each entry in CONFIGS overrides some of them.
DEFAULTS = dict(
    walks_per_node=100, depth=8, vector_size=200, window=5, epochs=10,
    graph="ours",  # "ours": the preprocessed graph; "authors": full_triple_set.csv with nodes identified by text, as their code reads it
    # direction: "forward" follows edges as stored; "both" adds an inverse edge for every edge between two
    #   entities, so a walk ends at the first literal it reaches; "through_literals" adds an inverse edge for
    #   every edge. In both two-way modes a walk never steps straight back to the node it came from, except
    #   with return_at_dead_end when that is the only way on.
    direction="forward", return_at_dead_end=False,
    start="all",  # "all": walks start at every node, as the authors do; "entities": at entities only
    flood=None,  # None (no geography), "all" (the authors' flooding) or "relation_aware" (E2)
    weights=None,  # None (uniform), "minmax" (our fix of the authors' min-max weights) or "authors_minmax"
    degree_alpha=0.0,  # E4: edge weight times degree(target) ** -alpha
    drop_hubs=0,  # E3: remove the top k hubs and their edges (select_hubs)
    strip_lang=False,  # E5: drop language tags, so "x"@nl and "x" become one node
    geo_walks_per_node=None,  # E6: walks per entity with its own geometry; others keep walks_per_node
    length_rule=None,  # E7: None, "cap", "soft", "budget" or "unreliable" (see StopRule)
    length_km=None,  # E7: distance for the length rules; None = geo.SPREAD_FRACTION x the dataset's baseline spread
)
TL = dict(direction="through_literals", start="entities", return_at_dead_end=True)  # E1 (b)
GEO = dict(flood="all", weights="minmax")
CONFIGS = {
    # E0: uniform walks, no geography
    "rdf2vec_forward": dict(),
    "rdf2vec_both": dict(direction="both", start="entities"),
    "rdf2vec_through_literals": TL,
    # E0: GeoRDF2Vec as published, with only word2vec fixed (it trains on no data as committed)
    "geo_published": dict(graph="authors", flood="all", weights="authors_minmax"),
    # Base and E1: our graph, the authors' flooding, our min-max weights, three walk types
    "geo_forward": GEO,
    "geo_both": {**GEO, "direction": "both", "start": "entities"},
    "geo_through_literals": {**GEO, **TL},
    # E2-E7, on E1 (b) walks with geography
    "geo_relation_aware": {**TL, "flood": "relation_aware", "weights": "minmax"},
    "geo_hubs10": {**GEO, **TL, "drop_hubs": 10},
    "geo_hubs50": {**GEO, **TL, "drop_hubs": 50},
    "geo_degree05": {**GEO, **TL, "degree_alpha": 0.5},
    "geo_degree1": {**GEO, **TL, "degree_alpha": 1.0},
    "geo_nolang": {**GEO, **TL, "strip_lang": True},
    "geo_walks50": {**GEO, **TL, "geo_walks_per_node": 50},
    "geo_walks200": {**GEO, **TL, "geo_walks_per_node": 200},
    "geo_len_cap": {**GEO, **TL, "length_rule": "cap"},
    "geo_len_soft": {**GEO, **TL, "length_rule": "soft"},
    "geo_len_budget": {**GEO, **TL, "length_rule": "budget"},
    "geo_len_unreliable": {**GEO, **TL, "length_rule": "unreliable"},
}
C_GRID = [0.001, 0.01, 0.1, 1, 10]
CHUNK_STARTS = 20_000  # start nodes per batch of walks, to bound memory


class Graph:
    """The walk graph: out-edges in CSR form. Nodes and relations are integer ids.

    fwd_* hold the triples (one edge per triple, as stored); src/dst/rel the edges a walk can take, sorted
    by source, and order[i] the position of CSR edge i in the forward edges followed by the inverse ones.
    """

    def __init__(self, keys, src, dst, rel, rel_names, entity_object, is_entity, direction):
        n = len(keys)
        self.keys, self.is_entity = keys, is_entity
        self.fwd_src, self.fwd_dst, self.fwd_rel = src, dst, rel
        self.fwd_rel_names = list(rel_names)
        inverse = {"forward": np.zeros(len(src), bool), "both": entity_object, "through_literals": np.ones(len(src), bool)}[direction]
        s, d = np.concatenate([src, dst[inverse]]), np.concatenate([dst, src[inverse]])
        r = np.concatenate([rel, rel[inverse] + len(rel_names)])
        self.rel_names = self.fwd_rel_names + ([] if direction == "forward" else ["inv:" + p for p in rel_names])
        self.order = np.argsort(s, kind="stable")
        self.src, self.dst, self.rel = s[self.order], d[self.order], r[self.order]
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(s, minlength=n))])
        self.total_degree = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)

    @classmethod
    def from_triples(cls, triples: pd.DataFrame, direction: str) -> "Graph":
        keys, inverse = np.unique(np.concatenate([triples["subject"].to_numpy(), triples["object"].to_numpy()]), return_inverse=True)
        n_triples = len(triples)
        src, dst = inverse[:n_triples], inverse[n_triples:]
        predicates, rel = np.unique(triples["predicate"].to_numpy(), return_inverse=True)
        entity_object = triples["object_kind"].eq(ENTITY).to_numpy()
        is_entity = np.zeros(len(keys), bool)
        is_entity[src] = True  # subjects are always entities
        is_entity[dst[entity_object]] = True
        return cls(keys, src, dst, rel, predicates, entity_object, is_entity, direction)

    @classmethod
    def from_authors_csv(cls, path: Path, direction: str) -> "Graph":
        """The graph as original/geo_rdf2vec.py builds it (igraph TupleList, nodes identified by text), with igraph's vertex numbers."""
        graph = build_graph(pd.read_csv(path))
        keys = np.array(graph.vs["name"], dtype=object)
        edges = np.array(graph.get_edgelist())
        predicates, rel = np.unique(np.array(graph.es["predicate"], dtype=object), return_inverse=True)
        is_entity = np.zeros(len(keys), bool)
        is_entity[edges[:, 0]] = True
        return cls(keys, edges[:, 0], edges[:, 1], rel, predicates, np.zeros(len(edges), bool), is_entity, direction)

    def degree(self, nodes: np.ndarray) -> np.ndarray:
        return self.indptr[nodes + 1] - self.indptr[nodes]


@dataclass
class StopRule:
    """E7. A walk ends before stepping to a node that breaks the rule ("cap", "soft", "budget"),
    or right after stepping onto an unreliable node ("unreliable").

    cap: the node is more than km from the start. soft: with probability 1 - exp(-distance from start / km).
    budget: the walk's summed step lengths would exceed km. unreliable: the node is near the national centroid.
    Distances involving a node without a location count as 0.
    """

    rule: str
    km: float
    lon: np.ndarray
    lat: np.ndarray
    unreliable: np.ndarray


def random_walks(graph: Graph, starts: np.ndarray, counts: np.ndarray, depth: int, rng: np.random.Generator, *,
                 no_backtrack: bool, return_at_dead_end: bool, prefix: np.ndarray | None, stop: StopRule | None):
    """Random walks; counts[i] walks from starts[i]. With prefix (cumulative edge weights, length E+1) the next
    edge is drawn in proportion to its weight, otherwise uniformly.
    Returns node ids (walks, depth+1) and relation ids (walks, depth); -1 after a walk ends."""

    def draw(nodes):
        low, high = graph.indptr[nodes], graph.indptr[nodes + 1]
        u = rng.random(len(nodes))
        uniform = low + (u * (high - low)).astype(np.int64)
        if prefix is None:
            return uniform
        lo, hi = prefix[low], prefix[high]
        picked = np.clip(np.searchsorted(prefix, lo + u * (hi - lo), side="right") - 1, low, high - 1)
        return np.where(hi > lo, picked, uniform)

    start = np.repeat(starts, counts)
    cur, prev = start.copy(), np.full(len(start), -1)
    alive = np.ones(len(start), bool)
    stop_next = np.zeros(len(start), bool)
    travelled = np.zeros(len(start))
    nodes, rels = [cur.copy()], []
    for _ in range(depth):
        alive &= (graph.degree(cur) > 0) & ~stop_next
        edge = np.zeros(len(cur), np.int64)
        edge[alive] = draw(cur[alive])
        if no_backtrack:
            # Redraw walks that chose the node they came from; after 10 tries, assume it is the only way on.
            for _ in range(10):
                back = alive & (graph.dst[edge] == prev)
                if not back.any():
                    break
                edge[back] = draw(cur[back])
            if not return_at_dead_end:
                alive &= graph.dst[edge] != prev
        nxt = graph.dst[edge]
        if stop is not None and stop.rule != "unreliable":
            a = np.flatnonzero(alive)
            if stop.rule == "budget":
                step = np.nan_to_num(geodesic_km(stop.lon[cur[a]], stop.lat[cur[a]], stop.lon[nxt[a]], stop.lat[nxt[a]]))
                end = travelled[a] + step > stop.km
                travelled[a] += step
            else:
                far = np.nan_to_num(geodesic_km(stop.lon[start[a]], stop.lat[start[a]], stop.lon[nxt[a]], stop.lat[nxt[a]]))
                end = far > stop.km if stop.rule == "cap" else rng.random(len(a)) < 1 - np.exp(-far / stop.km)
            alive[a[end]] = False
        if stop is not None and stop.rule == "unreliable":
            stop_next = alive & stop.unreliable[nxt]
        nodes.append(np.where(alive, nxt, -1))
        rels.append(np.where(alive, graph.rel[edge], -1))
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


def select_hubs(triples: pd.DataFrame, labelled: pd.Index, k: int) -> tuple[list, list]:
    """The k highest-degree nodes whose removal keeps every labelled node in the graph (E3).

    A labelled node is never removed, and a hub is skipped if every remaining edge of some labelled node
    goes through it. Returns the hubs removed and the hubs skipped."""
    keys, ends = np.unique(np.concatenate([triples["subject"], triples["object"]]), return_inverse=True)
    src, dst = ends[: len(triples)], ends[len(triples):]
    degree = np.bincount(ends, minlength=len(keys))
    is_labelled = pd.Index(keys).isin(labelled)
    remaining = degree.copy()  # edges left per node
    alive = np.ones(len(triples), bool)
    removed, skipped = [], []
    for node in np.argsort(-degree, kind="stable"):
        if len(removed) == k:
            break
        if is_labelled[node]:
            skipped.append(keys[node])
            continue
        edges = np.flatnonzero(alive & ((src == node) | (dst == node)))
        other = np.where(src[edges] == node, dst[edges], src[edges])
        lost = np.bincount(other, minlength=len(keys))
        if (is_labelled & (remaining > 0) & (remaining - lost <= 0)).any():
            skipped.append(keys[node])
            continue
        remaining -= lost
        remaining[node] = 0
        alive[edges] = False
        removed.append(keys[node])
    return removed, skipped


def load_triples(processed: Path, params: dict, info: dict) -> pd.DataFrame:
    triples = pd.read_parquet(processed / "triples.parquet")
    if params["drop_hubs"]:
        labelled = pd.Index(pd.read_csv(processed / "labels.csv")["node"])
        hubs, skipped = select_hubs(triples, labelled, params["drop_hubs"])
        info["hubs_skipped"] = [str(h) for h in skipped]
        triples = triples[~triples["subject"].isin(hubs) & ~triples["object"].isin(hubs)].reset_index(drop=True)
    if params["strip_lang"]:
        tagged = triples["object_kind"].str.startswith("@")
        triples.loc[tagged, "object"] = triples.loc[tagged, "object"].str.replace(r"@[A-Za-z0-9-]+$", "", regex=True)
        triples.loc[tagged, "object_kind"] = "plain"
    return triples


def flood_graph(graph: Graph, params: dict, processed: Path, cache_dir: Path, info: dict) -> geo.Flooded:
    """Flood geometries over the graph (cached: flooding does not depend on the seed or the walk direction)."""
    variant = f"{params['graph']}_{params['flood']}_hubs{params['drop_hubs']}_lang{int(params['strip_lang'])}"
    if params["drop_hubs"]:
        variant += "_keeplabelled"  # select_hubs; caches without it removed labelled hubs
    if params["flood"] == "relation_aware":
        variant += f"_spread{geo.SPREAD_FRACTION}_sharing{geo.MAX_SHARING}"
    cache = cache_dir / f"{variant}.npz"
    if cache.exists():
        c = np.load(cache)
        flooded = geo.Flooded(c["lon"], c["lat"], c["n_sources"], int(c["n_seeds"]))
    else:
        if params["graph"] == "authors":
            # Their case-sensitive expression, on str(label) as in their extract_geometry: pandas reads some values as NaN.
            seeds = sp.find_wkt_nodes([str(k) for k in graph.keys], sp.REPO_WKT_PATTERN)
            geometries = sp.parse_geometries(sp.extract_wkt(str(graph.keys[i])) for i in seeds)
        else:
            table = pd.read_parquet(processed / "geometries.parquet")
            ids = pd.Series(np.arange(len(graph.keys)), index=graph.keys).reindex(table["node"])
            table = table[ids.notna().to_numpy()]
            seeds = ids.dropna().to_numpy(np.int64)
            geometries = geo.seed_geometries_from_wkt(table["wkt"])
        keep = np.ones(len(graph.fwd_src), bool)
        if params["flood"] == "relation_aware":
            passes = set(spatial_relation_table(processed).query("passes")["relation"])
            keep = np.isin(np.array(graph.fwd_rel_names, dtype=object)[graph.fwd_rel], list(passes))
        flooded = geo.flood(len(graph.keys), graph.fwd_src[keep], graph.fwd_dst[keep], seeds, geometries)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, lon=flooded.lon, lat=flooded.lat, n_sources=flooded.n_sources, n_seeds=flooded.n_seeds)
    info.update(flood_variant=variant, seeds=flooded.n_seeds, located_share=float(np.mean(~np.isnan(flooded.lon))),
                unreliable_share=float(np.mean(flooded.unreliable)))
    return flooded


def spatial_relation_table(processed: Path) -> pd.DataFrame:
    triples = pd.read_parquet(processed / "triples.parquet")
    locations = geo.own_locations(triples, pd.read_parquet(processed / "geometries.parquet"))
    return geo.spatial_relations(pd.read_csv(processed / "relations.csv"), geo.baseline_spread_km(locations))


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
        params.update(walks_per_node=2, epochs=1, geo_walks_per_node=params["geo_walks_per_node"] and 4)
    name = f"{config}_seed{seed}" + ("_smoke" if smoke else "")
    result_path = RESULTS_DIR / dataset / f"{name}.json"
    if result_path.exists():
        print(f"{result_path.relative_to(REPO_ROOT)} exists, skipping")
        return
    data_dir = DATA_DIRS[dataset]
    processed = data_dir / "processed"
    run_dir = data_dir / "runs" / config
    run_dir.mkdir(parents=True, exist_ok=True)
    timings, info = {}, {}

    t = time.time()
    if params["graph"] == "authors":
        graph = Graph.from_authors_csv(data_dir / "full_triple_set.csv", params["direction"])
        triples = None
    else:
        triples = load_triples(processed, params, info)
        graph = Graph.from_triples(triples, params["direction"])
    starts = np.arange(len(graph.keys)) if params["start"] == "all" else np.flatnonzero(graph.is_entity)
    counts = np.full(len(starts), params["walks_per_node"])
    if params["geo_walks_per_node"]:
        located = geo.own_locations(triples, pd.read_parquet(processed / "geometries.parquet")).index
        has_own = pd.Index(graph.keys[starts]).isin(located)
        counts[has_own] = params["geo_walks_per_node"]
        info["starts_with_own_geometry"] = int(has_own.sum())
    timings["graph_s"] = time.time() - t

    t = time.time()
    prefix, stop = None, None
    if params["flood"]:
        flooded = flood_graph(graph, params, processed, data_dir / "runs" / "_flood", info)
        weights = np.ones(len(graph.dst))
        if params["weights"] == "minmax":
            weights = geo.minmax_weights(geo.edge_distances_km(flooded.lon, flooded.lat, graph.src, graph.dst), graph.indptr)
        elif params["weights"] == "authors_minmax":
            assert params["direction"] == "forward", "the authors' weights are defined on forward edges"
            d = geo.edge_distances_km(flooded.lon, flooded.lat, graph.fwd_src, graph.fwd_dst, swap_axes=True)
            weights = geo.authors_minmax_weights(d, graph.fwd_src, graph.fwd_dst, len(graph.keys))[graph.order]
        if params["length_rule"]:
            km = params["length_km"]
            if km is None:
                own = geo.own_locations(triples, pd.read_parquet(processed / "geometries.parquet"))
                km = geo.SPREAD_FRACTION * geo.baseline_spread_km(own)
            info["length_km"] = float(km)
            stop = StopRule(params["length_rule"], km, flooded.lon, flooded.lat, flooded.unreliable)
    else:
        weights = np.ones(len(graph.dst))
    if params["degree_alpha"]:
        weights = weights * graph.total_degree[graph.dst].astype(float) ** -params["degree_alpha"]
    if params["flood"] or params["degree_alpha"]:
        prefix = np.concatenate([[0.0], np.cumsum(weights)])
    timings["geography_s"] = time.time() - t

    t = time.time()
    rng = np.random.default_rng(seed)
    walk_file = run_dir / f"{name}.walks.txt"
    tokens, steps = 0, []
    with walk_file.open("w") as handle:
        for i in range(0, len(starts), CHUNK_STARTS):
            nodes, rels = random_walks(graph, starts[i : i + CHUNK_STARTS], counts[i : i + CHUNK_STARTS], params["depth"], rng,
                                       no_backtrack=params["direction"] != "forward", return_at_dead_end=params["return_at_dead_end"],
                                       prefix=prefix, stop=stop)
            steps.append(np.bincount((rels >= 0).sum(1), minlength=params["depth"] + 1))
            tokens += write_walks(nodes, rels, handle)
    steps = np.sum(steps, 0)
    info["mean_walk_steps"] = float((steps * np.arange(len(steps))).sum() / steps.sum())
    info["full_length_share"] = float(steps[-1] / steps.sum())
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
    # A labelled node can lose all its edges when hubs are removed (E3). It then gets a zero vector:
    # the embedding knows nothing about it.
    has_vector = [tok is not None and tok in model.wv.key_to_index for tok in tokens_of]
    info["labelled_without_vector"] = int(len(has_vector) - sum(has_vector))
    if info["labelled_without_vector"] > 0.01 * len(has_vector):
        raise RuntimeError(f"{info['labelled_without_vector']} labelled nodes have no vector")
    zero = np.zeros(model.wv.vector_size, dtype=np.float32)
    X = np.stack([model.wv[tok] if ok else zero for tok, ok in zip(tokens_of, has_vector)])
    np.savez_compressed(run_dir / f"{name}.npz", node=labels["node"].to_numpy(), vectors=X)

    t = time.time()
    metrics = evaluate(dataset, X, labels["class"].astype(str).to_numpy(), labels["split"].to_numpy(), seed)
    timings["classifier_s"] = time.time() - t

    result = {"dataset": dataset, "config": config, "seed": seed, "params": params, "smoke": smoke,
              "machine": platform.node(), "platform": platform.platform(), "cpus": os.cpu_count(),
              "start_nodes": int(len(starts)), "tokens": int(tokens), "info": info, "timings": timings, "validation": metrics}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"{dataset} {name}: macro F1 {metrics['macro_f1']:.3f}, accuracy {metrics['accuracy']:.3f}, "
          f"word2vec {timings['word2vec_s'] / 60:.1f} min, {tokens / 1e6:.0f}M tokens", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", choices=sorted(DATA_DIRS))
    parser.add_argument("configs", nargs="*", help="one or more of --list")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true", help="2 walks per node and 1 epoch, to test the pipeline end to end")
    parser.add_argument("--list", action="store_true", help="print the configurations and exit")
    args = parser.parse_args()
    if args.list:
        for name, overrides in CONFIGS.items():
            print(f"{name:26s} {overrides}")
        return
    unknown = set(args.configs) - set(CONFIGS)
    if not args.dataset or not args.configs or unknown:
        parser.error(f"give a dataset and configurations from --list{f'; unknown: {sorted(unknown)}' if unknown else ''}")
    for config in args.configs:
        for seed in args.seeds:
            run(args.dataset, config, seed, args.smoke)


if __name__ == "__main__":
    main()
