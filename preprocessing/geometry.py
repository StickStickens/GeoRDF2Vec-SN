"""Geometries: which nodes carry one, where it is, and distances between places.

The authors' code finds geometries with a case-sensitive regular expression over node
text and passes coordinates to geopy as (longitude, latitude), which geopy reads as
(latitude, longitude). Here a geometry is any literal typed as a WKT literal, parsed by
shapely (which accepts any letter case), and coordinates are always (longitude, latitude).
"""

import re

import numpy as np
import pandas as pd
import shapely
from pyproj import Geod

from .graph import node_key
from .load import WKT

_GEOD = Geod(ellps="WGS84")
_CRS_PREFIX = re.compile(r"^\s*<[^>]*>\s*")  # optional CRS IRI in front of a GeoSPARQL WKT literal
# Copied from GeoRDF2Vec.extract_geometry in original/geo_rdf2vec.py
AUTHORS_WKT_PATTERN = re.compile(
    r"(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)(\s?\(.+)(\))"
)


def geodesic_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Distance on the WGS84 ellipsoid, in kilometres. Scalars broadcast against arrays.

    Based on geodesic_km in eda/spatial.py.
    """
    lon1, lat1, lon2, lat2 = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (lon1, lat1, lon2, lat2)))
    _, _, metres = _GEOD.inv(lon1, lat1, lon2, lat2)
    return np.asarray(metres) / 1000


def normalise_wkt(text: pd.Series) -> pd.Series:
    """Write WKT keywords in capitals, e.g. 'Point(5 52)' as 'POINT(5 52)'.

    WKT ignores letter case, so the geometry stays the same, but every reader that searches
    node text, such as the authors' expression, now finds it. Only words of two or more
    letters change, so numbers such as 5e-5 and a CRS IRI prefix are kept as they are.
    """
    parts = text.str.extract(r"^(\s*<[^>]*>\s*)?(.*)$", flags=re.DOTALL)
    keywords = parts[1].str.replace(r"[A-Za-z]{2,}", lambda m: m.group(0).upper(), regex=True)
    return parts[0].fillna("") + keywords


def parse_wkt(text: pd.Series) -> np.ndarray:
    """Shapely geometries; None where the text is not valid WKT."""
    return shapely.from_wkt(text.str.replace(_CRS_PREFIX, "", regex=True).to_numpy(), on_invalid="ignore")


def extract_geometries(triples: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One row per distinct WKT literal that holds a usable geometry: node key, WKT, type and centroid.

    Also returns counts of the WKT literals that were left out and why.
    """
    text = pd.Series(triples.loc[triples["object_kind"].eq(WKT), "object"].unique())
    geometry = parse_wkt(text)
    invalid = pd.isna(geometry)
    empty = shapely.is_empty(geometry)  # False for None
    usable = ~invalid & ~empty
    centroid = shapely.centroid(geometry[usable])
    table = pd.DataFrame(
        {
            "node": node_key(text[usable], pd.Series(WKT, index=text[usable].index)).to_numpy(),
            "wkt": text[usable].to_numpy(),
            "geometry_type": shapely.get_type_id(geometry[usable]),
            "lon": shapely.get_x(centroid),
            "lat": shapely.get_y(centroid),
        }
    )
    type_names = {0: "Point", 1: "LineString", 3: "Polygon", 4: "MultiPoint", 5: "MultiLineString", 6: "MultiPolygon", 7: "GeometryCollection"}
    table["geometry_type"] = table["geometry_type"].map(type_names)
    counts = pd.Series({"WKT literals": len(text), "Not valid WKT": int(invalid.sum()), "Empty": int(empty.sum()), "Usable": int(usable.sum())})
    return table, counts


def authors_detection_count(texts: pd.Series) -> int:
    """How many of these texts the authors' regular expression recognises as a geometry."""
    return int(texts.map(lambda t: AUTHORS_WKT_PATTERN.search(t) is not None).sum())


def own_locations(triples: pd.DataFrame, geometries: pd.DataFrame) -> pd.DataFrame:
    """Location of each entity that has its own geometry: the mean centroid of its WKT literals.

    An entity has its own geometry if it links to a WKT literal directly, or through a
    geometry node (GeoSPARQL hasGeometry, then a WKT literal), as dmg777k monuments do.
    Every other node only gets a location through flooding, which is part of the experiments.
    Re-implements the "own geometry" step of the EDA notebooks.
    """
    centroids = geometries.set_index("node")[["lon", "lat"]]  # triples must use node keys
    direct = triples.loc[triples["object"].isin(centroids.index), ["subject", "object"]]
    via_node = triples.loc[triples["predicate"].str.endswith("hasGeometry"), ["subject", "object"]].merge(
        direct.rename(columns={"subject": "object", "object": "wkt"}), on="object"
    )
    links = pd.concat([direct.rename(columns={"object": "wkt"}), via_node[["subject", "wkt"]]]).drop_duplicates()
    located = links.join(centroids, on="wkt")
    return located.groupby("subject")[["lon", "lat"]].mean()
