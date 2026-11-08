"""Read BGS+ as prepared by data_preparation/bgs_data_prep.py."""

from pathlib import Path

import pandas as pd

from ..paths import REPO_ROOT

BGS_DIR = REPO_ROOT / "data" / "bgs"
WKT_KIND = "http://www.opengis.net/ont/geosparql#wktLiteral"


def load_triples(data_dir: Path = BGS_DIR) -> pd.DataFrame:
    """Subject, predicate, object text, in the form geo_rdf2vec.py reads."""
    return pd.read_csv(data_dir / "full_triple_set.csv", dtype=str, keep_default_na=False)


def load_node_kinds(data_dir: Path = BGS_DIR) -> pd.Series:
    """Kind per node text: 'entity', or the literal's datatype IRI or language tag."""
    return pd.read_csv(data_dir / "node_kinds.csv", dtype=str, keep_default_na=False).set_index("label")["kind"]


def load_labels(data_dir: Path = BGS_DIR) -> pd.DataFrame:
    """Labelled rock units: node text, class (FLUV or GLACI) and split."""
    return pd.read_csv(data_dir / "labels.csv")


def kind_group(kind: str) -> str:
    """Collapse a node kind into the groups used in the analysis."""
    if kind == "entity":
        return "Entity (IRI)"
    if kind.startswith("@"):
        return "Language-tagged text"
    short = kind.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return {
        "wktLiteral": "WKT geometry",
        "XMLLiteral": "XML text",
        "MapCode": "Map code",
        "string": "Untyped text",
        "plain": "Untyped text",
        "anyURI": "URI string",
        "decimal": "Number",
        "integer": "Number",
        "int": "Number",
        "date": "Date",
    }.get(short, short)


def entity_type(iri: str) -> str:
    """Short type of a BGS entity from its IRI path, e.g. 'Lexicon/NamedRockUnit'."""
    for marker in ("/id/Lexicon/", "/id/625KGeologyMap/", "/id/EarthMaterialClass/", "/id/Geochronology/"):
        if marker in iri:
            return marker.strip("/").split("/")[-1] + "/" + iri.split(marker)[1].split("/")[0]
    return "other"
