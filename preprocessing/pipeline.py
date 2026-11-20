"""Run the whole preprocessing for one dataset and write the results to data/<dataset>/processed/.

    uv run python -m preprocessing.pipeline dmg777k
    uv run python -m preprocessing.pipeline bgs
"""

import argparse
from dataclasses import dataclass

import pandas as pd
import shapely

from . import catalog, geometry, graph, labels
from .load import LOADERS, WKT, Dataset

# Two points with a known distance, written in both letter cases the data uses.
AMSTERDAM_UTRECHT = ("POINT(4.90 52.37)", "Point(5.12 52.09)", 34.6)  # km, from geopy with (lat, lon)


@dataclass
class Result:
    dataset: Dataset
    placeholders_removed: pd.DataFrame
    identity: pd.DataFrame
    triples: pd.DataFrame  # subject, predicate, object (node keys), object_kind
    geometries: pd.DataFrame  # node, wkt, geometry_type, lon, lat
    geometry_counts: pd.Series
    locations: pd.DataFrame  # entity -> lon, lat of its own geometry
    relations: pd.DataFrame
    hubs: pd.DataFrame
    splits: pd.DataFrame
    label_checks: pd.DataFrame
    checks: pd.DataFrame


def fix_checks(dataset: Dataset, raw_wkt: pd.Series, identity: pd.DataFrame, geometries: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """Section 1.2: each problem the EDA found in the authors' pipeline, before and after our preprocessing.

    raw_wkt holds the WKT texts as the dataset writes them, before normalise_wkt.
    """
    not_capitals = lambda texts: int((texts != geometry.normalise_wkt(texts)).sum())  # noqa: E731
    lon_min, lat_min, lon_max, lat_max = dataset.bbox
    outside = ~geometries["lon"].between(lon_min, lon_max) | ~geometries["lat"].between(lat_min, lat_max)
    a, b, expected = AMSTERDAM_UTRECHT
    pa, pb = geometry.parse_wkt(pd.Series([a, b]))
    ours = geometry.geodesic_km(shapely.get_x(pa), shapely.get_y(pa), shapely.get_x(pb), shapely.get_y(pb))
    # The authors pass (lon, lat) to geopy, which reads it as (lat, lon): longitude and latitude swap places.
    swapped = geometry.geodesic_km(shapely.get_y(pa), shapely.get_x(pa), shapely.get_y(pb), shapely.get_x(pb))
    rows = [
        ("WKT literals recognised as geometries", geometry.authors_detection_count(raw_wkt), int(counts["Usable"]), f"all usable of {counts['WKT literals']:,}"),
        ("WKT literals with keywords not in capitals", not_capitals(raw_wkt), not_capitals(geometries["wkt"]), "0"),
        ("Geometry centres outside the country", None, int(outside.sum()), "0"),
        (f"Amsterdam–Utrecht distance, km (true {expected})", round(float(swapped), 1), round(float(ours), 1), f"{expected}"),
        ("Self-loops", int(identity.loc["Self-loops"].iloc[0]), int(identity.loc["Self-loops"].iloc[1]), "0"),
        ("Nodes merging an entity and a literal", int(identity.iloc[2, 0]), int(identity.iloc[2, 1]), "0"),
        ("Nodes merging literals of different types", int(identity.iloc[3, 0]), int(identity.iloc[3, 1]), "0"),
    ]
    return pd.DataFrame(rows, columns=["check", "authors' pipeline", "ours", "expected"]).set_index("check")


def run(name: str) -> Result:
    dataset = LOADERS[name]()
    text_triples, removed = graph.drop_placeholders(dataset.triples)
    is_wkt = text_triples["object_kind"].eq(WKT)
    raw_wkt = pd.Series(text_triples.loc[is_wkt, "object"].unique())
    text_triples.loc[is_wkt, "object"] = geometry.normalise_wkt(text_triples.loc[is_wkt, "object"])
    identity = graph.identity_report(text_triples)
    triples = graph.keyed_triples(text_triples)
    geometries, counts = geometry.extract_geometries(text_triples)
    locations = geometry.own_locations(triples, geometries)
    return Result(
        dataset=dataset,
        placeholders_removed=removed,
        identity=identity,
        triples=triples,
        geometries=geometries,
        geometry_counts=counts,
        locations=locations,
        relations=catalog.relation_table(triples, locations),
        hubs=catalog.hub_table(triples),
        splits=labels.split_table(dataset.labels),
        label_checks=labels.label_checks(dataset.labels, triples, dataset.target_relations),
        checks=fix_checks(dataset, raw_wkt, identity, geometries, counts),
    )


def save(result: Result) -> None:
    out = result.dataset.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    result.triples.to_parquet(out / "triples.parquet", index=False)
    result.geometries.to_parquet(out / "geometries.parquet", index=False)
    result.dataset.labels.to_csv(out / "labels.csv", index=False)
    result.relations.to_csv(out / "relations.csv", index=False)
    result.hubs.to_csv(out / "hubs.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", choices=sorted(LOADERS))
    for name in parser.parse_args().datasets:
        result = run(name)
        save(result)
        print(f"{name}: {len(result.triples):,} triples, {len(result.geometries):,} geometries -> {result.dataset.processed_dir}")
        print(result.checks.to_string(), end="\n\n")


if __name__ == "__main__":
    main()
