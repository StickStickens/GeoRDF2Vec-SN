"""Turn BGS+ (N-Triples) into the triple CSV that geo_rdf2vec.py reads, plus label and node-kind tables.

BGS+ stores geometries in British National Grid metres (EPSG:27700) without a CRS tag.
geo_rdf2vec.py treats every WKT as WGS84 longitude/latitude, so all geometries are
reprojected to EPSG:4326 here, as dmg_data_prep.py does for the RD New geometries.

Outputs in data/bgs:
    full_triple_set.csv  subject, predicate, object (literals as their lexical form)
    labels.csv           node, class, split
    node_kinds.csv       label, kind (entity, or the literal's datatype / language tag)
"""

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "bgs"
SPLIT_FILES = {"train": "bgs+_train_set.nt.gz", "valid": "bgs+_test_valid.nt.gz", "test": "bgs+_test_set.nt.gz"}
WKT_TYPE = "http://www.opengis.net/ont/geosparql#wktLiteral"

TRIPLE = re.compile(r'^<([^>]*)>\s+<([^>]*)>\s+(.*?)\s*\.\s*$')
IRI_OBJECT = re.compile(r"^<([^>]*)>$")
LITERAL_OBJECT = re.compile(r'^"(.*)"(?:\^\^<([^>]*)>|@([A-Za-z-]+))?$', re.DOTALL)
ESCAPES = re.compile(r'\\(["\\nrt])')
ESCAPE_MAP = {'"': '"', "\\": "\\", "n": "\n", "r": "\r", "t": "\t"}


def parse_ntriples(path: Path) -> pd.DataFrame:
    """Subject, predicate, object text and object kind for every triple in a gzipped N-Triples file."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            match = TRIPLE.match(line)
            if match is None:
                raise ValueError(f"{path.name}:{line_number}: cannot parse {line[:120]!r}")
            subject, predicate, obj = match.groups()
            if iri := IRI_OBJECT.match(obj):
                rows.append((subject, predicate, iri[1], "entity"))
                continue
            literal = LITERAL_OBJECT.match(obj)
            if literal is None:
                raise ValueError(f"{path.name}:{line_number}: cannot parse object {obj[:120]!r}")
            text, datatype, language = literal.groups()
            text = ESCAPES.sub(lambda m: ESCAPE_MAP[m[1]], text)
            kind = datatype or (f"@{language}" if language else "plain")
            rows.append((subject, predicate, text, kind))
    return pd.DataFrame(rows, columns=["subject", "predicate", "object", "kind"])


def reproject_wkt(wkt: pd.Series) -> pd.Series:
    """British National Grid WKT to WGS84 WKT with (longitude, latitude) axis order."""
    to_wgs84 = Transformer.from_crs(27700, 4326, always_xy=True)
    geometries = shapely.from_wkt(wkt.to_numpy())
    reprojected = shapely.transform(geometries, lambda xy: np.column_stack(to_wgs84.transform(xy[:, 0], xy[:, 1])))
    return pd.Series(shapely.to_wkt(reprojected, rounding_precision=7), index=wkt.index)


def load_labels() -> pd.DataFrame:
    frames = []
    for split, name in SPLIT_FILES.items():
        triples = parse_ntriples(DATA_DIR / name)
        frames.append(pd.DataFrame({"node": triples["subject"], "class": triples["object"].str.rsplit("/", n=1).str[-1], "split": split}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    triples = parse_ntriples(DATA_DIR / "bgs+.nt.gz")
    is_wkt = triples["kind"].eq(WKT_TYPE)
    triples.loc[is_wkt, "object"] = reproject_wkt(triples.loc[is_wkt, "object"])

    kinds = pd.concat(
        [
            pd.DataFrame({"label": triples["subject"], "kind": "entity"}),
            triples[["object", "kind"]].rename(columns={"object": "label"}),
        ]
    )
    # A text used both as an entity and as a literal becomes one node; count it as an entity.
    kinds["is_entity"] = kinds["kind"].eq("entity")
    kinds = kinds.sort_values("is_entity", ascending=False).drop_duplicates("label")[["label", "kind"]]

    triples[["subject", "predicate", "object"]].to_csv(DATA_DIR / "full_triple_set.csv", index=False)
    kinds.to_csv(DATA_DIR / "node_kinds.csv", index=False)
    labels = load_labels()
    labels.to_csv(DATA_DIR / "labels.csv", index=False)
    print(f"{len(triples):,} triples, {len(kinds):,} distinct nodes, {is_wkt.sum():,} WKT literals reprojected")
    print(labels.groupby(["split", "class"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
