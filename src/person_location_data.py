from __future__ import annotations

from pathlib import Path

import polars as pl

from storage_paths import PROJECT_ROOT, RAW_DATA_DIR

PREDICATE = "VISITED"


def read_csv(path: Path) -> pl.DataFrame:
    """Read a raw node or relationship CSV as a Polars DataFrame.

    Keeping CSV as the source of truth makes the graph easy to grow: add rows,
    rerun graph ingestion, then rerun HDC encoding against the same LanceDB data.
    """
    return pl.read_csv(path, infer_schema=False)


def person_records(raw_data_dir: Path = RAW_DATA_DIR) -> pl.DataFrame:
    """Load Person node rows from raw CSV."""
    return read_csv(raw_data_dir / "nodes" / "person.csv")


def location_records(raw_data_dir: Path = RAW_DATA_DIR) -> pl.DataFrame:
    """Load Location node rows from raw CSV, reading each image in natively.

    A multimodal knowledge graph stores its assets, not just pointers to them.
    `image_path` stays as a human-readable reference, while `image` holds the
    raw bytes of that file so the skyline photo lives in the same LanceDB row
    as the graph facts and hypervectors, versioned and queried together.
    """
    locations = read_csv(raw_data_dir / "nodes" / "location.csv")
    image_bytes = [
        (PROJECT_ROOT / image_path).read_bytes()
        for image_path in locations.get_column("image_path")
    ]
    return locations.with_columns(pl.Series("image", image_bytes, dtype=pl.Binary))


def relationship_records(raw_data_dir: Path = RAW_DATA_DIR) -> pl.DataFrame:
    """Load VISITED relationship rows from raw CSV."""
    return read_csv(raw_data_dir / "relationships" / "visited.csv")


def location_vibe_records(raw_data_dir: Path = RAW_DATA_DIR) -> pl.DataFrame:
    """Load raw multimodal vibe evidence rows for Location nodes."""
    return read_csv(raw_data_dir / "features" / "location_vibes.csv")


def validate_relationships(
    persons: pl.DataFrame,
    locations: pl.DataFrame,
    relationships: pl.DataFrame,
) -> None:
    """Fail early when relationship CSV rows point at missing nodes."""
    missing_people = (
        relationships.select("person_id")
        .join(persons.select(pl.col("id").alias("person_id")), on="person_id", how="anti")
        .get_column("person_id")
        .unique()
        .sort()
        .to_list()
    )
    missing_locations = (
        relationships.select("location_id")
        .join(
            locations.select(pl.col("id").alias("location_id")),
            on="location_id",
            how="anti",
        )
        .get_column("location_id")
        .unique()
        .sort()
        .to_list()
    )
    if missing_people or missing_locations:
        raise ValueError(
            "Relationship CSV references missing nodes: "
            f"people={missing_people}, locations={missing_locations}"
        )


def validate_location_vibes(locations: pl.DataFrame, vibes: pl.DataFrame) -> None:
    """Fail early when vibe evidence references missing locations or bad weights."""
    missing_locations = (
        vibes.select("location_id")
        .join(
            locations.select(pl.col("id").alias("location_id")),
            on="location_id",
            how="anti",
        )
        .get_column("location_id")
        .unique()
        .sort()
        .to_list()
    )
    bad_weights = (
        vibes.with_columns(pl.col("weight").cast(pl.Float64, strict=False).alias("_weight"))
        .filter(pl.col("_weight").is_null() | (pl.col("_weight") < 0) | (pl.col("_weight") > 1))
        .select("location_id", "feature", "weight")
        .to_dicts()
    )
    if missing_locations or bad_weights:
        raise ValueError(
            "Location vibe CSV contains invalid rows: "
            f"locations={missing_locations}, weights={bad_weights}"
        )
