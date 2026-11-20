"""Read each dataset into one triple table that keeps the type of every object.

The authors' pipeline keeps only the text of each node. We also keep its kind: "entity"
for an IRI, or for a literal its datatype IRI, its language tag ("@nl"), or "plain".
Subjects are always entities in both datasets.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

ENTITY = "entity"
PLAIN = "plain"
WKT = "http://www.opengis.net/ont/geosparql#wktLiteral"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"

DMG_DIR = DATA_DIR / "kgbench_dmg777k"
DMG_SPLITS = {"train": "training", "valid": "validation", "test": "testing"}
DMG_RD_PREDICATE = "http://data.pdok.nl/def/pdok#asWKT-RD"

BGS_DIR = DATA_DIR / "bgs"


@dataclass
class Dataset:
    name: str
    triples: pd.DataFrame  # subject, predicate, object, object_kind
    labels: pd.DataFrame  # node, class, split
    target_relations: tuple[str, ...]  # removed from the graph by the dataset authors
    bbox: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max of the covered country
    processed_dir: Path


def normalise_kind(kind: pd.Series) -> pd.Series:
    """In RDF 1.1 a plain literal and an xsd:string literal are the same value."""
    return kind.replace({XSD_STRING: PLAIN, "none": PLAIN, "iri": ENTITY})


def reproject_wkt(wkt: pd.Series, source_epsg: int) -> pd.Series:
    """WKT in a projected CRS to WGS84 WKT in (longitude, latitude) order. Unparseable text becomes None.

    Based on reproject_wkt in data/bgs_data_prep.py, with the source CRS as a parameter.
    """
    to_wgs84 = Transformer.from_crs(source_epsg, 4326, always_xy=True)
    geometries = shapely.from_wkt(wkt.to_numpy(), on_invalid="ignore")
    reprojected = shapely.transform(geometries, lambda xy: np.column_stack(to_wgs84.transform(xy[:, 0], xy[:, 1])))
    return pd.Series(shapely.to_wkt(reprojected), index=wkt.index)


def load_dmg777k(data_dir: Path = DMG_DIR) -> Dataset:
    """dmg777k from the raw KGBench files, with the authors' RD New to WGS84 conversion.

    Follows original/data_preparation/dmg_data_prep.py and the loaders in eda/dmg777k/data.py.
    """
    nodes = pd.read_csv(data_dir / "nodes.int.csv", dtype=str, keep_default_na=False).astype({"index": int})
    nodes = nodes.set_index("index")
    relations = pd.read_csv(data_dir / "relations.int.csv").set_index("index")["label"]
    raw = pd.read_csv(data_dir / "triples.int.csv.gz", header=None, names=["s", "p", "o"], skipinitialspace=True)

    kind = normalise_kind(nodes["annotation"])
    triples = pd.DataFrame(
        {
            "subject": raw["s"].map(nodes["label"]).to_numpy(),
            "predicate": raw["p"].map(relations).to_numpy(),
            "object": raw["o"].map(nodes["label"]).to_numpy(),
            "object_kind": raw["o"].map(kind).to_numpy(),
        }
    )
    assert raw["s"].map(kind).eq(ENTITY).all(), "every subject should be an entity"

    # As original/data_preparation/dmg_data_prep.py does: 371 municipality outlines are stored
    # in RD New metres (EPSG:28992) as untyped text. Convert them to WGS84 WKT literals.
    is_rd = triples["predicate"].eq(DMG_RD_PREDICATE)
    triples.loc[is_rd, "object"] = reproject_wkt(triples.loc[is_rd, "object"], 28992)
    triples.loc[is_rd, "object_kind"] = WKT
    triples = triples.dropna(subset=["object"]).reset_index(drop=True)

    labels = pd.concat(
        [pd.read_csv(data_dir / f"{stem}.int.csv").assign(split=split) for split, stem in DMG_SPLITS.items()],
        ignore_index=True,
    )
    labels = pd.DataFrame({"node": labels["index"].map(nodes["label"]), "class": labels["class"], "split": labels["split"]})

    return Dataset(
        name="dmg777k",
        triples=triples,
        labels=labels,
        target_relations=("monumentCode", "http://purl.org/dc/terms/subject"),
        bbox=(3.2, 50.7, 7.3, 53.6),
        processed_dir=data_dir / "processed",
    )


def load_bgs(data_dir: Path = BGS_DIR) -> Dataset:
    """BGS+ as written by data/bgs_data_prep.py, which already reprojects British National Grid to WGS84."""
    path = data_dir / "full_triple_set.csv"
    triples = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "object_kind" not in triples.columns:
        raise ValueError(f"{path} has no object_kind column; run `uv run python data/bgs_data_prep.py` again")
    triples["object_kind"] = normalise_kind(triples["object_kind"])
    labels = pd.read_csv(data_dir / "labels.csv")
    return Dataset(
        name="bgs",
        triples=triples,
        labels=labels,
        target_relations=("hasLithogenesis",),
        bbox=(-11.0, 49.0, 2.5, 61.5),
        processed_dir=data_dir / "processed",
    )


LOADERS = {"dmg777k": load_dmg777k, "bgs": load_bgs}
