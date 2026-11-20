"""Descriptive tables for the experiments: one row per relation, and the hubs.

They contain facts only. Which relations pass on geometries, and what to do with hubs,
is decided in the experiments.
"""

import pandas as pd

from .geometry import geodesic_km
from .load import ENTITY


def spatial_spread(triples: pd.DataFrame, locations: pd.DataFrame, min_subjects: int = 30) -> pd.DataFrame:
    """How close together are the located subjects that share an object, per relation.

    For each (relation, object) shared by at least two located subjects, each subject's
    distance to the group's mean position. A small median means the relation links things
    that lie close together, e.g. a street; a large one means it does not, e.g. a class.
    Relations with fewer than min_subjects such subjects get no value.

    Based on relation_dispersion in eda/spatial.py (same measure).
    """
    rows = triples.join(locations, on="subject", how="inner")
    groups = rows.groupby(["predicate", "object"])
    rows = rows[groups["lon"].transform("size") >= 2]
    groups = rows.groupby(["predicate", "object"])
    distance = geodesic_km(rows["lon"], rows["lat"], groups["lon"].transform("mean"), groups["lat"].transform("mean"))
    rows = rows.assign(distance_km=distance)
    spread = rows.groupby("predicate").agg(
        located_subjects=("distance_km", "size"), median_km=("distance_km", "median")
    )
    spread.loc[spread["located_subjects"] < min_subjects, "median_km"] = float("nan")
    return spread


def relation_table(triples: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """One row per relation: size, what it points to, and its spatial spread."""
    is_entity = triples["object_kind"].eq(ENTITY)
    table = triples.groupby("predicate").agg(
        edges=("object", "size"),
        distinct_objects=("object", "nunique"),
        entity_share=("object_kind", lambda k: k.eq(ENTITY).mean()),
    )
    literal_kinds = triples[~is_entity].groupby("predicate")["object_kind"].agg(lambda k: k.value_counts().index[0])
    table["object"] = table["entity_share"].map(lambda s: "entity" if s == 1 else "literal" if s == 0 else "mixed")
    table["literal_kind"] = literal_kinds
    table = table.join(spatial_spread(triples, locations))
    table["median_km"] = table["median_km"].round(2)
    table = table.drop(columns="entity_share").rename_axis("relation").reset_index()
    return table.sort_values("edges", ascending=False, ignore_index=True)[
        ["relation", "edges", "distinct_objects", "object", "literal_kind", "located_subjects", "median_km"]
    ]


def hub_table(triples: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    """The nodes with the most edges, the kind of node, and the relation that brings most of them."""
    out_degree = triples["subject"].value_counts()
    in_degree = triples["object"].value_counts()
    degree = out_degree.add(in_degree, fill_value=0).astype(int)
    hubs = degree.nlargest(top).rename("degree").to_frame()
    hubs["in_degree"] = in_degree.reindex(hubs.index, fill_value=0)
    hubs["out_degree"] = out_degree.reindex(hubs.index, fill_value=0)
    object_kind = triples.drop_duplicates("object").set_index("object")["object_kind"]
    hubs["kind"] = object_kind.reindex(hubs.index).where(~hubs.index.isin(out_degree.index), ENTITY)

    ends = pd.concat(
        [
            triples.loc[triples["subject"].isin(hubs.index), ["subject", "predicate"]].set_axis(["node", "predicate"], axis=1),
            triples.loc[triples["object"].isin(hubs.index), ["object", "predicate"]].set_axis(["node", "predicate"], axis=1),
        ]
    )
    main = ends.groupby(["node", "predicate"]).size().rename("n").reset_index()
    main = main.sort_values("n", ascending=False).drop_duplicates("node").set_index("node")
    hubs["main_relation"] = main["predicate"]
    hubs["main_relation_share"] = (main["n"] / hubs["degree"]).round(3)
    return hubs.rename_axis("node").reset_index()[
        ["node", "kind", "degree", "in_degree", "out_degree", "main_relation", "main_relation_share"]
    ]
