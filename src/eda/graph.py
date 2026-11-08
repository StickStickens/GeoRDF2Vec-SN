"""Build the graph exactly as the repository's GeoRDF2Vec pipeline does."""

import pandas as pd
from igraph import Graph


def build_graph(triples: pd.DataFrame) -> Graph:
    """Build the graph exactly as GeoRDF2Vec.read_kg_file does: nodes are identified by text."""
    records = triples[["subject", "object", "predicate"]].to_records(index=False)
    return Graph().TupleList(edges=records, directed=True, edge_attrs=["predicate"])
