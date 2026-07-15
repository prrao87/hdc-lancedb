from __future__ import annotations

import argparse
import json
from pathlib import Path

import lancedb

from graph_retrieve import build_engine, city_query
from hdc_encode import prepare_vibes
from person_location_data import location_vibe_records
from storage_paths import DEFAULT_DB_URI, DEFAULT_VOCAB_PATH
from torchhd_encoder import TorchHDEncoder


def load_encoder(vocab_path: Path = DEFAULT_VOCAB_PATH) -> TorchHDEncoder:
    """Load the same token vocabulary used during HDC encoding."""
    return TorchHDEncoder(json.loads(vocab_path.read_text()))


def query_vibe_features(query: str) -> list[str]:
    """Map a small natural-language query family onto multimodal vibe features."""
    normalized = query.lower()
    if "pacific coast" in normalized and "mountain" in normalized:
        return ["pacific_coast", "mountains", "nature_access", "waterfront"]
    if "mountain city vibes" in normalized:
        return ["mountains", "scenic_urban", "nature_access"]
    if "places with mountains" in normalized or "mountains" in normalized:
        return ["mountains", "nature_access"]
    if "concrete jungle" in normalized:
        return ["dense_skyline", "concrete_jungle", "urban_energy"]
    raise ValueError(
        "Unsupported fuzzy query. Try 'persons from cities on the pacific coast "
        "with mountains nearby', 'places with mountains', 'mountain city vibes', "
        "or 'concrete jungle'."
    )


def query_vibe_hv(query: str, encoder: TorchHDEncoder):
    """Encode desired features with the same role/value bindings as stored rows."""
    return encoder.bundle(
        [
            encoder.association_hv("feature", feature)
            for feature in query_vibe_features(query)
        ]
    )


def vibe_features_by_location() -> dict[str, list[dict[str, str]]]:
    """Load explanatory vibe features, strongest first, grouped by location."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for vibe in prepare_vibes(location_vibe_records()):
        grouped.setdefault(vibe["location_id"], []).append(vibe)
    for rows in grouped.values():
        rows.sort(key=lambda row: (-float(row["weight"]), row["feature"]))
    return grouped


def hdc_fuzzy_paths(
    query: str,
    db_uri: Path = DEFAULT_DB_URI,
    vocab_path: Path = DEFAULT_VOCAB_PATH,
    threshold: float = 0.20,
    limit: int = 25,
) -> list[dict[str, object]]:
    """Search location vibes in LanceDB, then traverse matching graph paths."""
    if limit <= 0:
        raise ValueError("LanceDB search limit must be positive")
    encoder = load_encoder(vocab_path)
    db = lancedb.connect(db_uri)
    vibes = vibe_features_by_location()

    query_location_hv = query_vibe_hv(query, encoder)
    location_rows = (
        db.open_table("Location")
        .search(
            query_location_hv.detach().cpu().tolist(),
            vector_column_name="vibe_hv",
        )
        .distance_type("cosine")
        .select(["id", "name", "timezone", "_distance"])
        .limit(limit)
        .to_arrow()
        .to_pylist()
    )
    engine = build_engine(db_uri)

    matches = []
    for location in location_rows:
        score = 1.0 - float(location["_distance"])
        if score < threshold:
            continue
        top_features = [
            row["feature"] for row in vibes.get(location["id"], [])[:3]
        ]
        for path in engine.execute(city_query(location["name"])).to_pylist():
            matches.append(
                {
                    "person": path["person"],
                    "city": path["city"],
                    "timezone": location["timezone"],
                    "features": top_features,
                    "score": score,
                }
            )

    return sorted(matches, key=lambda row: (-row["score"], row["person"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search HDC vectors in LanceDB, then traverse matching graph paths."
    )
    parser.add_argument("--db-uri", type=Path, default=DEFAULT_DB_URI)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument(
        "--query",
        required=True,
        help=(
            "Fuzzy natural-language query, e.g. 'persons from cities on the "
            "pacific coast with mountains nearby'."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Similarity threshold for fuzzy HDC retrieval.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of locations returned by LanceDB vector search.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for row in hdc_fuzzy_paths(
        args.query,
        args.db_uri,
        args.vocab_path,
        threshold=args.threshold,
        limit=args.limit,
    ):
        print({**row, "score": round(row["score"], 3)})
