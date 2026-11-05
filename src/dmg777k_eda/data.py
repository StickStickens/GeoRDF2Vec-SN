"""Read dmg777k into tables and into the same igraph graph the repo's pipeline uses."""

from pathlib import Path

import pandas as pd
from igraph import Graph

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "kgbench_dmg777k"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
SPLITS = {"train": "training", "valid": "validation", "test": "testing"}


def load_raw_nodes(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Raw node table: index, annotation, label. Empty labels stay empty strings."""
    return pd.read_csv(data_dir / "nodes.int.csv", dtype=str, keep_default_na=False).astype({"index": int})


def load_relations(data_dir: Path = DATA_DIR) -> pd.Series:
    """Relation IRI by integer ID."""
    return pd.read_csv(data_dir / "relations.int.csv").set_index("index")["label"]


def load_raw_triples(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Raw integer triples, before the repo's preparation step."""
    return pd.read_csv(
        data_dir / "triples.int.csv.gz", header=None, names=["s", "p", "o"], skipinitialspace=True
    )


def load_prepared_triples(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Output of data_preparation/dmg_data_prep.py, read the way geo_rdf2vec.py reads it."""
    return pd.read_csv(data_dir / "full_triple_set.csv")


def build_graph(triples: pd.DataFrame) -> Graph:
    """Build the graph exactly as GeoRDF2Vec.read_kg_file does: nodes are identified by text."""
    records = triples[["subject", "object", "predicate"]].to_records(index=False)
    return Graph().TupleList(edges=records, directed=True, edge_attrs=["predicate"])


def load_labels(raw_nodes: pd.DataFrame, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Labelled monuments with their text identifier, class and split."""
    frames = [
        pd.read_csv(data_dir / f"{stem}.int.csv").assign(split=split)
        for split, stem in SPLITS.items()
    ]
    labels = pd.concat(frames, ignore_index=True)
    labels["node"] = labels["index"].map(raw_nodes.set_index("index")["label"])
    return labels


def annotation_group(annotation: str) -> str:
    """Collapse an annotation value into the node kinds used in the analysis."""
    if annotation == "iri":
        return "Entity (IRI)"
    if annotation == "none":
        return "Untyped text"
    if annotation.startswith("@"):
        return "Language-tagged text"
    short = annotation.rsplit("#", 1)[-1]
    return {
        "base64Image": "Image",
        "wktLiteral": "WKT geometry",
        "anyURI": "URI string",
        "nonNegativeInteger": "Number",
        "positiveInteger": "Number",
        "gYear": "Year",
        "boolean": "Boolean",
    }[short]


def merged_node_is_entity(raw_nodes: pd.DataFrame) -> pd.Series:
    """Per distinct text: True if any raw node with that text is an entity."""
    return raw_nodes["annotation"].eq("iri").groupby(raw_nodes["label"]).any()
