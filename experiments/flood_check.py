"""E2 without labels: hide known locations, flood again, and measure how far off flooding puts them.

    uv run python -m experiments.flood_check dmg777k bgs

dmg777k: 20% of the monuments with their own geometry lose their hasGeometry links, per fold.
BGS+: 20% of the rock units with a stratotype lose their hasStratotype links, per fold; the truth is the
mean location of their stratotypes. 5 folds, so every such node is hidden once.
Writes experiments/results/<dataset>/flood_check.csv: one row per flooding variant.

    uv run python -m experiments.flood_check dmg777k bgs --sweep

also floods relation-aware at other spread fractions (geo.SPREAD_FRACTION), into flood_sweep.csv.
"""

import argparse

import numpy as np
import pandas as pd

from experiments import geo
from experiments.run import DATA_DIRS, RESULTS_DIR, Graph
from preprocessing.geometry import geodesic_km

FOLDS = 5


def targets(dataset: str, triples: pd.DataFrame, own: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Nodes to hide with their true location, and per triple the target it links (NaN if none)."""
    if dataset == "dmg777k":
        links = triples["predicate"].str.endswith("#hasGeometry")
        truth = own.loc[own.index.intersection(triples.loc[links, "subject"].unique())]
        owner = triples["subject"].where(links)
    else:
        # Only hasStratotype: isStratotypeFor points to separate .../ref/Lexicon/NamedRockUnit/ nodes, not to the
        # .../id/ units that carry the labels. As in the EDA's hold-out.
        links = triples["predicate"].str.endswith("hasStratotype")
        located = triples.loc[links, ["subject", "object"]].join(own, on="object", how="inner")
        truth = located.groupby("subject")[["lon", "lat"]].mean()
        owner = triples["subject"].where(links)
    return truth, owner


SWEEP = [0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0]


def check(dataset: str, sweep: bool = False) -> pd.DataFrame:
    processed = DATA_DIRS[dataset] / "processed"
    triples = pd.read_parquet(processed / "triples.parquet")
    table = pd.read_parquet(processed / "geometries.parquet")
    own = geo.own_locations(triples, table)
    relations, baseline = pd.read_csv(processed / "relations.csv"), geo.baseline_spread_km(own)
    truth, owner = targets(dataset, triples, own)
    graph = Graph.from_triples(triples, "forward")
    node_id = pd.Series(np.arange(len(graph.keys)), index=graph.keys)
    seeds = node_id.reindex(table["node"]).to_numpy(np.int64)
    seed_geometries = geo.seed_geometries_from_wkt(table["wkt"])

    def kept(fraction: float) -> tuple[frozenset, np.ndarray]:
        passes = frozenset(geo.spatial_relations(relations, baseline, fraction).query("passes")["relation"])
        return passes, triples["predicate"].isin(passes).to_numpy()

    variants = {"all relations (authors)": (None, np.ones(len(triples), bool)),
                "relation-aware": (geo.SPREAD_FRACTION, kept(geo.SPREAD_FRACTION)[1])}
    if sweep:  # one variant per distinct set of relations
        seen = {}
        for fraction in SWEEP:
            passes, mask = kept(fraction)
            seen.setdefault(passes, (fraction, mask))
        variants = {f"relation-aware {f}": (f, mask) for f, mask in seen.values()}
    folds = np.random.default_rng(0).permutation(len(truth)) % FOLDS
    print(f"{dataset}: {len(truth):,} nodes to hide, {FOLDS} folds")

    rows = []
    for variant, (fraction, keep) in variants.items():
        errors, centroid, unlocated = [], 0, 0
        for fold in range(FOLDS):
            hidden = truth.index[folds == fold]
            cut = owner.isin(hidden).to_numpy()
            mask = keep & ~cut
            flooded = geo.flood(len(graph.keys), graph.fwd_src[mask], graph.fwd_dst[mask], seeds, seed_geometries)
            ids = node_id.reindex(hidden).to_numpy(np.int64)
            t = truth.loc[hidden]
            error = geodesic_km(t["lon"].to_numpy(), t["lat"].to_numpy(), flooded.lon[ids], flooded.lat[ids])
            errors.append(error[~np.isnan(error)])
            unlocated += int(np.isnan(error).sum())
            centroid += int(flooded.unreliable[ids].sum())
        errors = np.concatenate(errors)
        rows.append({"dataset": dataset, "variant": variant, "spread_fraction": fraction,
                     "relations": int(len(set(triples.loc[keep, "predicate"]))), "hidden": len(truth), "median_km": np.median(errors),
                     "p90_km": np.quantile(errors, 0.9), "at_national_centroid": centroid / len(truth),
                     "without_location": unlocated / len(truth)})
        print(f"  {variant:24s} median {rows[-1]['median_km']:7.1f} km, 90th percentile {rows[-1]['p90_km']:7.1f} km, "
              f"at the national centroid {rows[-1]['at_national_centroid']:.1%}, no location {rows[-1]['without_location']:.1%}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / dataset / ("flood_sweep.csv" if sweep else "flood_check.csv"), index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", choices=sorted(DATA_DIRS))
    parser.add_argument("--sweep", action="store_true", help="relation-aware flooding at several spread fractions")
    args = parser.parse_args()
    for dataset in args.datasets:
        check(dataset, args.sweep)


if __name__ == "__main__":
    main()
