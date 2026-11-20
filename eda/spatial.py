"""Geometry parsing, the repo's flooding and edge weighting, and spatial statistics."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from igraph import Graph
from pyproj import Geod
from scipy import sparse

_GEOD = Geod(ellps="WGS84")
_WKT_TYPES = "POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION"
REPO_WKT_PATTERN = re.compile(rf"({_WKT_TYPES})(\s?\(.+)(\))")
ANY_CASE_WKT_PATTERN = re.compile(rf"({_WKT_TYPES})(\s?\(.+)(\))", re.IGNORECASE)
NL_BBOX = (3.2, 50.7, 7.3, 53.6)


def geodesic_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Ellipsoidal WGS84 distance in kilometres."""
    return _GEOD.inv(lon1, lat1, lon2, lat2)[2] / 1000.0


def find_wkt_nodes(names: Sequence[str], pattern: re.Pattern) -> np.ndarray:
    """Vertex IDs whose text matches pattern, mirroring GeoRDF2Vec.extract_geometry."""
    return np.array([i for i, name in enumerate(names) if pattern.search(name)], dtype=np.int64)


def extract_wkt(text: str) -> str:
    """Text of the first geometry in a label; the pattern is case-insensitive."""
    return ANY_CASE_WKT_PATTERN.search(text)[0]


def parse_geometries(wkt: Iterable[str]) -> np.ndarray:
    """Shapely geometries from WKT strings of any length."""
    return shapely.from_wkt(np.array(list(wkt), dtype=object))


@dataclass
class FloodResult:
    """Outcome of flooding, with identical source sets stored once.

    hops is -1 where flooding did not reach a vertex, as is set_id. sets[set_id[v]]
    holds the positions in the seed list of the geometries vertex v inherits.
    """

    hops: np.ndarray
    set_id: np.ndarray
    sets: list[np.ndarray]

    @property
    def n_sources(self) -> np.ndarray:
        sizes = np.array([len(members) for members in self.sets] + [0])
        return sizes[self.set_id]


def undirected_adjacency(graph: Graph) -> sparse.csr_matrix:
    edges = np.array(graph.get_edgelist())
    n = graph.vcount()
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    return sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))


def flood(graph: Graph, seeds: np.ndarray) -> FloodResult:
    """Multi-source BFS over undirected edges, equivalent to GeoRDF2Vec.neighborhood_flood.

    A vertex first reached in round k inherits every seed at distance exactly k.
    Many vertices inherit the same set, so unions are computed once per distinct input.
    """
    adjacency = undirected_adjacency(graph)
    n = graph.vcount()
    hops = np.full(n, -1, dtype=np.int32)
    set_id = np.full(n, -1, dtype=np.int32)
    sets = [np.array([position], dtype=np.int32) for position in range(len(seeds))]
    hops[seeds] = 0
    set_id[seeds] = np.arange(len(seeds))
    known_sets: dict[bytes, int] = {members.tobytes(): i for i, members in enumerate(sets)}
    union_cache: dict[bytes, int] = {}

    frontier = seeds
    round_number = 0
    while len(frontier):
        round_number += 1
        reached = np.unique(adjacency[frontier].indices)
        new_nodes = reached[hops[reached] == -1]
        for node in new_nodes:
            neighbours = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
            inputs = np.unique(set_id[neighbours[hops[neighbours] == round_number - 1]])
            if len(inputs) == 1:
                set_id[node] = inputs[0]
                continue
            key = inputs.tobytes()
            if key not in union_cache:
                members = np.unique(np.concatenate([sets[i] for i in inputs]))
                member_key = members.tobytes()
                if member_key not in known_sets:
                    known_sets[member_key] = len(sets)
                    sets.append(members)
                union_cache[key] = known_sets[member_key]
            set_id[node] = union_cache[key]
        hops[new_nodes] = round_number
        frontier = new_nodes
    return FloodResult(hops, set_id, sets)


def collection_centroid(geometries: np.ndarray) -> tuple[float, float]:
    """Centroid as GeoRDF2Vec computes it: a lone geometry's own, else its GeometryCollection's."""
    if len(geometries) == 1:
        centroid = shapely.centroid(geometries[0])
    else:
        centroid = shapely.centroid(shapely.geometrycollections(geometries))
    return float(shapely.get_x(centroid)), float(shapely.get_y(centroid))


def node_centroids(flooded: FloodResult, geometries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude per vertex; NaN where flooding did not reach the vertex."""
    centroids = np.array([collection_centroid(geometries[members]) for members in flooded.sets])
    reached = flooded.set_id >= 0
    x, y = np.full(len(reached), np.nan), np.full(len(reached), np.nan)
    x[reached] = centroids[flooded.set_id[reached], 0]
    y[reached] = centroids[flooded.set_id[reached], 1]
    return x, y


def edge_distances(graph: Graph, x: np.ndarray, y: np.ndarray, swap_axes: bool) -> np.ndarray:
    """Distance per edge in km; 0 where a centroid is missing, as in the repo.

    swap_axes=True reproduces the repo: it passes (lon, lat) to geopy, which reads (lat, lon).
    """
    edges = np.array(graph.get_edgelist())
    u, v = edges[:, 0], edges[:, 1]
    if swap_axes:
        distances = geodesic_km(y[u], x[u], y[v], x[v])
    else:
        distances = geodesic_km(x[u], y[u], x[v], y[v])
    return np.nan_to_num(distances, nan=0.0)


def repo_minmax_weights(graph: Graph, distances: np.ndarray) -> np.ndarray:
    """Edge weights as computed in GeoRDF2Vec.kg_training with min_max_normalization.

    The repo loops over vertices and overwrites the weight of each incident edge,
    so an edge keeps the value computed for its higher-numbered endpoint.
    """
    edges = np.array(graph.get_edgelist())
    frame = pd.DataFrame(
        {"node": np.concatenate([edges[:, 0], edges[:, 1]]), "distance": np.tile(distances, 2)}
    )
    grouped = frame.groupby("node")["distance"]
    low, high = grouped.min(), grouped.max()
    owner = edges.max(axis=1)
    owner_low = low.reindex(owner).to_numpy()
    span = high.reindex(owner).to_numpy() - owner_low
    scaled = np.divide(distances - owner_low, span, out=np.zeros_like(distances), where=span > 0)
    return np.exp(-scaled)


def repo_vertex_loop_weights(graph: Graph, distances: np.ndarray) -> np.ndarray:
    """Slow, literal version of the repo's vertex loop, for checking repo_minmax_weights."""
    weights = np.ones(graph.ecount())
    for vertex in range(graph.vcount()):
        incident = np.array(graph.incident(vertex, mode="all"))
        if len(incident) == 0:
            continue
        values = distances[incident]
        low, high = values.min(), values.max()
        scaled = (values - low) / (high - low) if high > low else np.zeros(len(values))
        weights[incident] = np.exp(-scaled)
    return weights


def relation_dispersion(triples: pd.DataFrame, x: pd.Series, y: pd.Series, min_rows: int = 200) -> pd.DataFrame:
    """How tightly a shared object groups its located subjects, per relation.

    For each (predicate, object) with at least two located subjects, measures each
    subject's distance to the group's mean position. A small value means the relation
    links things that are close together. Inputs x and y are indexed by node text.
    """
    located = triples[triples["subject"].isin(x.index)].copy()
    located["lon"] = located["subject"].map(x)
    located["lat"] = located["subject"].map(y)
    grouped = located.groupby(["predicate", "object"])
    located["group_size"] = grouped["lon"].transform("size")
    located = located[located["group_size"] >= 2].copy()
    grouped = located.groupby(["predicate", "object"])
    located["distance"] = geodesic_km(
        located["lon"].to_numpy(),
        located["lat"].to_numpy(),
        grouped["lon"].transform("mean").to_numpy(),
        grouped["lat"].transform("mean").to_numpy(),
    )
    summary = located.groupby("predicate").agg(
        rows=("distance", "size"),
        groups=("object", "nunique"),
        q25=("distance", lambda d: d.quantile(0.25)),
        median=("distance", "median"),
        q75=("distance", lambda d: d.quantile(0.75)),
    )
    return summary[summary["rows"] >= min_rows].sort_values("median")


def outgoing_tables(graph: Graph, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-edge lookup for walks: index pointers, targets, cumulative weights per vertex."""
    edges = np.array(graph.get_edgelist())
    order = np.argsort(edges[:, 0], kind="stable")
    sources, targets = edges[order, 0], edges[order, 1]
    step_weights = np.ones(len(order)) if weights is None else weights[order]
    pointers = np.searchsorted(sources, np.arange(graph.vcount() + 1))
    return pointers, targets, step_weights


def simulate_walks(
    graph: Graph,
    starts: np.ndarray,
    steps: int,
    weights: np.ndarray | None,
    walks_per_start: int,
    seed: int,
    undirected: bool = False,
) -> list[list[int]]:
    """Walks following igraph's random_walk defaults: outgoing edges only, stop at a dead end.

    With undirected=True every edge can also be walked backwards, with the same weight.
    """
    if undirected:
        edges = np.array(graph.get_edgelist())
        graph = Graph(n=graph.vcount(), edges=np.vstack([edges, edges[:, ::-1]]), directed=True)
        weights = None if weights is None else np.concatenate([weights, weights])
    pointers, targets, step_weights = outgoing_tables(graph, weights)
    rng = np.random.default_rng(seed)
    walks = []
    for start in starts:
        for _ in range(walks_per_start):
            walk = [int(start)]
            current = int(start)
            for _ in range(steps):
                low, high = pointers[current], pointers[current + 1]
                if low == high:
                    break
                cumulative = np.cumsum(step_weights[low:high])
                pick = np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")
                current = int(targets[low + min(pick, high - low - 1)])
                walk.append(current)
            walks.append(walk)
    return walks
