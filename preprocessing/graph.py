"""Node identity and missing values.

The authors' code builds the graph from node text alone, so an entity and a literal with
the same text, or two literals of different types, become one node. We identify a node by
its text and kind instead, which is how RDF identifies terms.
"""

import pandas as pd

from .load import ENTITY, PLAIN, WKT

# Texts that stand for "no value". Compared case-insensitively, without a trailing full stop.
PLACEHOLDERS = frozenset({"", "not entered", "not defined", "unknown", "not known", "not applicable", "n/a"})
# An empty WKT geometry, e.g. "POLYGON EMPTY", optionally after a CRS IRI: a geometry with no location.
EMPTY_WKT = r"^\s*(?:<[^>]*>\s*)?[A-Za-z]+\s+EMPTY\s*$"


def node_key(text: pd.Series, kind: pd.Series) -> pd.Series:
    """Unique node name: the IRI for an entity, an N-Triples-style term for a literal.

    Examples: '"1880"^^<http://www.w3.org/2001/XMLSchema#gYear>', '"kerk"@nl', '"36"'.
    Quotes and backslashes are escaped, so two different (text, kind) pairs never share a key.
    """
    quoted = '"' + text.str.replace("\\", "\\\\", regex=False).str.replace('"', '\\"', regex=False) + '"'
    suffix = pd.Series("", index=kind.index)
    tagged = kind.str.startswith("@")
    typed = ~tagged & kind.ne(PLAIN) & kind.ne(ENTITY)
    suffix[tagged] = kind[tagged]
    suffix[typed] = "^^<" + kind[typed] + ">"
    return text.where(kind.eq(ENTITY), quoted + suffix)


def is_placeholder(text: pd.Series) -> pd.Series:
    return text.str.strip().str.rstrip(".").str.lower().isin(PLACEHOLDERS)


def drop_placeholders(triples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove triples whose object is a literal meaning "no value": a placeholder text or an empty geometry.

    Returns the kept triples and what was removed.
    """
    empty_wkt = triples["object_kind"].eq(WKT) & triples["object"].str.match(EMPTY_WKT, case=False)
    missing = triples["object_kind"].ne(ENTITY) & (is_placeholder(triples["object"]) | empty_wkt)
    removed = (
        triples[missing]
        .groupby(["object", "predicate"])
        .size()
        .rename("triples")
        .reset_index()
        .sort_values("triples", ascending=False, ignore_index=True)
    )
    return triples[~missing].reset_index(drop=True), removed


def identity_report(triples: pd.DataFrame) -> pd.DataFrame:
    """Nodes, self-loops and wrong merges when nodes are identified by text (authors) or by text and kind (ours)."""
    terms = pd.concat(
        [
            pd.DataFrame({"text": triples["subject"], "kind": ENTITY}),
            triples[["object", "object_kind"]].set_axis(["text", "kind"], axis=1),
        ]
    ).drop_duplicates()
    terms["key"] = node_key(terms["text"], terms["kind"])

    def merges(name: str) -> tuple[int, int, int]:
        """Distinct nodes, merges into an entity, merges of literals of different types."""
        per_node = terms.groupby(name)["kind"].agg(kinds="nunique", has_entity=lambda k: k.eq(ENTITY).any())
        shared = per_node[per_node["kinds"] > 1]
        return len(per_node), int(shared["has_entity"].sum()), int((~shared["has_entity"]).sum())

    authors, ours = merges("text"), merges("key")
    loops_text = int(triples["subject"].eq(triples["object"]).sum())
    loops_key = int(triples["subject"].eq(node_key(triples["object"], triples["object_kind"])).sum())
    rows = {
        "Nodes": (authors[0], ours[0]),
        "Self-loops": (loops_text, loops_key),
        "Nodes where a literal is merged into an entity": (authors[1], ours[1]),
        "Nodes where literals of different types are merged": (authors[2], ours[2]),
    }
    return pd.DataFrame(rows, index=["By text (authors)", "By text and kind (ours)"]).T


def keyed_triples(triples: pd.DataFrame) -> pd.DataFrame:
    """Triples with node keys as subject and object, plus the object's kind."""
    return pd.DataFrame(
        {
            "subject": triples["subject"],
            "predicate": triples["predicate"],
            "object": node_key(triples["object"], triples["object_kind"]),
            "object_kind": triples["object_kind"],
        }
    )
