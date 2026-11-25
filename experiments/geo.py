"""Geography for the experiments: which relations pass on locations, flooding, and spatial edge weights.

Flooding reuses the EDA's re-implementation of the authors' neighborhood_flood (eda/spatial.py), which
the EDA checked against their code: a node first reached in round k inherits every geometry at distance
exactly k, and its location is the centroid of their GeometryCollection.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from igraph import Graph as IGraph

from eda import spatial as sp
from preprocessing.geometry import geodesic_km, own_locations
from preprocessing.load import WKT

# Relation-aware flooding (E2). A relation passes on locations if the located subjects sharing one of its
# objects lie close together: median distance to their group centre at most SPREAD_FRACTION of the
# dataset's baseline spread. Where that spread is unknown (too few located subjects), a relation passes
# if its objects are shared by at most MAX_SHARING subjects on average, so it cannot form a hub.
# 0.4 (30 km on dmg777k, 99 km on BGS+) reproduces the hand-made BGS+ list of the EDA, which its hold-out
# check validated: 0.5 keeps BGS+ ages, colours and map codes (106-123 km); 0.33 drops hasLexicon (85 km),
# which links the map to the rock units.
SPREAD_FRACTION = 0.4
MAX_SHARING = 20
# A node is "unreliable" (E7, rule 4) if flooding gave it more than this share of all geometries,
# which puts it near the national centroid.
UNRELIABLE_SHARE = 0.5


def baseline_spread_km(locations: pd.DataFrame) -> float:
    """Median distance of the entities with their own geometry to their mean position (EDA: 75 km on dmg777k)."""
    lon, lat = locations["lon"].to_numpy(), locations["lat"].to_numpy()
    return float(np.median(geodesic_km(lon, lat, lon.mean(), lat.mean())))


def spatial_relations(relations: pd.DataFrame, baseline_km: float, fraction: float = SPREAD_FRACTION) -> pd.DataFrame:
    """relations.csv with a column `passes`: whether the relation passes on locations in relation-aware flooding."""
    spread, sharing = relations["median_km"], relations["edges"] / relations["distinct_objects"]
    known = spread.notna()
    passes = (known & (spread <= fraction * baseline_km)) | (~known & (sharing <= MAX_SHARING))
    passes |= relations["literal_kind"].eq(WKT)  # the geometries themselves
    return relations.assign(sharing=sharing.round(1), passes=passes)


@dataclass
class Flooded:
    lon: np.ndarray  # per node; NaN where flooding did not reach the node
    lat: np.ndarray
    n_sources: np.ndarray  # geometries each node inherits
    n_seeds: int

    @property
    def unreliable(self) -> np.ndarray:
        return self.n_sources > UNRELIABLE_SHARE * self.n_seeds


def flood(n_nodes: int, src: np.ndarray, dst: np.ndarray, seeds: np.ndarray, seed_geometries: np.ndarray) -> Flooded:
    """Flood geometries from the seed nodes over the given edges, ignoring their direction."""
    graph = IGraph(n=n_nodes, edges=np.column_stack([src, dst]).tolist(), directed=True)
    result = sp.flood(graph, seeds)
    lon, lat = sp.node_centroids(result, seed_geometries)
    return Flooded(lon, lat, result.n_sources, len(seeds))


def edge_distances_km(lon: np.ndarray, lat: np.ndarray, src: np.ndarray, dst: np.ndarray, swap_axes: bool = False) -> np.ndarray:
    """Distance per edge; 0 where an end has no location, as in the authors' code.

    swap_axes reproduces the authors' bug: they pass (lon, lat) to geopy, which reads (lat, lon).
    """
    if swap_axes:
        d = geodesic_km(lat[src], lon[src], lat[dst], lon[dst])
    else:
        d = geodesic_km(lon[src], lat[src], lon[dst], lat[dst])
    return np.nan_to_num(d, nan=0.0)


def minmax_weights(distances: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    """Our fix of the authors' min_max_normalization: exp(-scaled distance), scaled to [0, 1] over the
    edges a walk at each node can take next. Every edge gets one weight for each direction it is walked in."""
    n = len(indptr) - 1
    source = np.repeat(np.arange(n), np.diff(indptr))
    low = np.full(n, np.inf)
    high = np.full(n, -np.inf)
    np.minimum.at(low, source, distances)
    np.maximum.at(high, source, distances)
    span = (high - low)[source]
    scaled = np.divide(distances - low[source], span, out=np.zeros_like(distances), where=span > 0)
    return np.exp(-scaled)


def authors_minmax_weights(distances: np.ndarray, src: np.ndarray, dst: np.ndarray, n_nodes: int) -> np.ndarray:
    """The authors' min_max_normalization as it runs: normalised over all edges at a node, and each edge
    keeps the value computed for its higher-numbered end, which their vertex loop writes last.
    Same as repo_minmax_weights in eda/spatial.py, on arrays. Vertex numbers must be igraph's."""
    nodes = np.concatenate([src, dst])
    both = np.tile(distances, 2)
    low = np.full(n_nodes, np.inf)
    high = np.full(n_nodes, -np.inf)
    np.minimum.at(low, nodes, both)
    np.maximum.at(high, nodes, both)
    owner = np.maximum(src, dst)
    span = high[owner] - low[owner]
    scaled = np.divide(distances - low[owner], span, out=np.zeros_like(distances), where=span > 0)
    return np.exp(-scaled)


def seed_geometries_from_wkt(wkt: pd.Series) -> np.ndarray:
    return shapely.from_wkt(wkt.str.replace(r"^\s*<[^>]*>\s*", "", regex=True).to_numpy())


__all__ = ["baseline_spread_km", "spatial_relations", "flood", "Flooded", "edge_distances_km", "minmax_weights",
           "authors_minmax_weights", "seed_geometries_from_wkt", "own_locations"]
