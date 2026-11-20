"""Labels and splits: counts and the checks that no label leaks into training."""

import pandas as pd


def split_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Labelled nodes per class and split."""
    table = labels.pivot_table(index="class", columns="split", values="node", aggfunc="count", fill_value=0)
    table = table[["train", "valid", "test"]]
    table["total"] = table.sum(axis=1)
    table.loc["total"] = table.sum()
    return table


def label_checks(labels: pd.DataFrame, triples: pd.DataFrame, target_relations: tuple[str, ...]) -> pd.DataFrame:
    """Each check with its count; every count should be 0. Re-implements the leakage checks of the EDA notebooks."""
    in_graph = set(triples["subject"]) | set(triples["object"])
    target = triples["predicate"].map(lambda p: any(t in p for t in target_relations))
    checks = {
        "Nodes labelled more than once": int(labels["node"].duplicated().sum()),
        "Labelled nodes missing from the graph": int((~labels["node"].isin(in_graph)).sum()),
        f"Triples using the target relation ({', '.join(target_relations)})": int(target.sum()),
    }
    return pd.Series(checks, name="count").to_frame()
