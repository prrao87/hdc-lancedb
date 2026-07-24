from __future__ import annotations

from pathlib import Path

import lancedb

from graph.data import location_evidence_records
from graph.paths import (
    DEFAULT_DB_URI,
    DEFAULT_OLLAMA_HOST,
    SEMANTIC_HDC_METADATA_FILENAME,
)
from graph.retrieve import build_engine, city_query
from hdc.encode import LocationEvidence, prepare_location_evidence
from hdc.semantic import (
    OllamaEmbedder,
    RandomHyperplaneProjector,
    SemanticHDCMetadata,
)


def load_semantic_runtime(
    db_uri: Path,
    ollama_host: str,
) -> tuple[OllamaEmbedder, RandomHyperplaneProjector, SemanticHDCMetadata]:
    """Load and validate the exact semantic coordinate system used at encode time."""
    metadata = SemanticHDCMetadata.read(
        db_uri / SEMANTIC_HDC_METADATA_FILENAME
    )
    embedder = OllamaEmbedder(model=metadata.model, host=ollama_host)
    metadata.validate_runtime(
        model=embedder.canonical_model,
        model_digest=embedder.model_digest(),
    )
    projector = RandomHyperplaneProjector.load(
        db_uri / metadata.projection_file,
        seed=metadata.projection_seed,
        expected_sha256=metadata.projection_sha256,
    )
    if (
        projector.input_dimensions != metadata.embedding_dimensions
        or projector.output_dimensions != metadata.hdc_dimensions
    ):
        raise ValueError("Projection artifact shape does not match semantic metadata")
    return embedder, projector, metadata


def semantic_features_by_location() -> dict[str, list[LocationEvidence]]:
    """Load explanatory semantic features, strongest first by location."""
    grouped: dict[str, list[LocationEvidence]] = {}
    for evidence in prepare_location_evidence(location_evidence_records()):
        grouped.setdefault(evidence["location_id"], []).append(evidence)
    for rows in grouped.values():
        rows.sort(key=lambda row: (-float(row["weight"]), row["feature"]))
    return grouped


def hdc_fuzzy_paths(
    query: str,
    db_uri: Path = DEFAULT_DB_URI,
    min_score: float | None = None,
    limit: int = 5,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> list[dict[str, object]]:
    """Search full Location nodes in LanceDB, then traverse matching graph paths."""
    if limit <= 0:
        raise ValueError("LanceDB search limit must be positive")
    if min_score is not None and not -1.0 <= min_score <= 1.0:
        raise ValueError("Minimum cosine score must be between -1 and 1")
    embedder, projector, metadata = load_semantic_runtime(db_uri, ollama_host)
    db = lancedb.connect(db_uri)
    features = semantic_features_by_location()

    query_location_hv = projector.project(embedder.embed_query(query))
    if query_location_hv.numel() != metadata.hdc_dimensions:
        raise ValueError("Query projection dimension does not match stored vectors")
    location_rows = (
        db.open_table("Location")
        .search(
            query_location_hv.detach().cpu().numpy(),
            vector_column_name="hv",
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
        if min_score is not None and score < min_score:
            continue
        top_features = [
            row["feature"] for row in features.get(location["id"], [])[:3]
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
