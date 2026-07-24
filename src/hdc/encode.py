from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, TypedDict

import lance
import polars as pl
import pyarrow as pa
import torch
import torchhd

from graph.data import (
    PREDICATE,
    location_evidence_records,
    validate_location_evidence,
)
from graph.paths import (
    DEFAULT_DB_URI,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    RAW_DATA_DIR,
    SEMANTIC_HDC_METADATA_FILENAME,
    SEMANTIC_PROJECTION_FILENAME,
)
from hdc.core import (
    DIMENSIONS,
    NON_SYMBOLIC_COLUMNS,
    SEED,
    TorchHDEncoder,
)
from hdc.location import DEFAULT_SEMANTIC_WEIGHT, LocationNodeEncoder
from hdc.semantic import (
    PROJECTION_TYPE,
    OllamaEmbedder,
    RandomHyperplaneProjector,
    SemanticHDCMetadata,
)


@dataclass(frozen=True)
class EncodedTable:
    """One batched float32 HDC column keyed by graph row ID."""

    ids: list[str]
    hvs: torch.Tensor

    def __post_init__(self) -> None:
        if self.hvs.ndim != 2 or self.hvs.shape != (
            len(self.ids),
            DIMENSIONS,
        ):
            raise ValueError(
                f"Invalid hv tensor shape {tuple(self.hvs.shape)} for "
                f"{len(self.ids)} rows"
            )
        if self.hvs.dtype != torch.float32:
            raise TypeError("hv must remain float32 until Arrow storage")


@dataclass(frozen=True)
class EncodedRows:
    """Table-wise HDC batches ready to merge into the graph dataset."""

    persons: EncodedTable
    locations: EncodedTable
    relationships: EncodedTable


class LocationEvidence(TypedDict):
    location_id: str
    feature: str
    weight: float
    evidence: str


def prepare_location_evidence(
    vibes: pl.DataFrame,
) -> list[LocationEvidence]:
    """Select and normalize evidence used by semantic HDC encoding."""
    prepared = []
    for row in vibes.iter_rows(named=True):
        prepared.append(
            {
                "location_id": row["location_id"],
                "feature": row["feature"],
                "weight": float(row["weight"]),
                "evidence": row["evidence"],
            }
        )
    return prepared


def build_location_documents(
    location_evidence: list[LocationEvidence],
) -> dict[str, str]:
    """Build one deterministic semantic retrieval document per location.

    Feature labels are converted from machine-friendly snake case to ordinary
    text and paired with their evidence. The numeric weight controls ordering,
    but is intentionally not embedded as a number: the model receives the
    actual semantic evidence rather than numeric weight tokens.
    """
    grouped: dict[str, list[LocationEvidence]] = {}
    for evidence in location_evidence:
        grouped.setdefault(evidence["location_id"], []).append(evidence)

    documents = {}
    for location_id, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (-float(row["weight"]), row["feature"], row["evidence"]),
        )
        clauses = [
            f"{row['feature'].replace('_', ' ')}: {row['evidence']}"
            for row in ordered
        ]
        documents[location_id] = "; ".join(clauses)
    return documents


def dataset_path(db_uri: Path, table_name: str) -> str:
    """Filesystem path of one Lance table inside the shared dataset."""
    return str(db_uri / f"{table_name}.lance")


def read_table(db_uri: Path, table_name: str) -> pl.DataFrame:
    """Read symbolic graph columns, skipping blobs and prior HDC vectors.

    Neither image bytes nor already-encoded vectors are inputs to a fresh HDC
    encoding pass. Projecting both out avoids materializing large columns and
    keeps repeat encoding proportional to the source graph properties.
    """
    ds = lance.dataset(dataset_path(db_uri, table_name))
    columns = [
        name for name in ds.schema.names if name not in NON_SYMBOLIC_COLUMNS
    ]
    return pl.from_arrow(ds.to_table(columns=columns))


def encode_rows(
    persons: pl.DataFrame,
    locations: pl.DataFrame,
    relationships: pl.DataFrame,
    location_semantic_hvs: Mapping[str, torchhd.MAPTensor],
    encoder: TorchHDEncoder,
    location_node_encoder: LocationNodeEncoder,
) -> EncodedRows:
    """Encode graph rows, delegating whole Location nodes to their façade."""
    person_hvs = {}
    person_ids = []
    person_vectors = []
    for person in persons.iter_rows(named=True):
        person_hv = encoder.encode_properties(person)
        person_hvs[person["id"]] = person_hv
        person_ids.append(person["id"])
        person_vectors.append(person_hv)

    location_hvs = {}
    location_ids = []
    location_vectors = []
    for location in locations.iter_rows(named=True):
        location_semantic_hv = location_semantic_hvs.get(location["id"])
        if location_semantic_hv is None:
            raise ValueError(f"Location has no semantic evidence: {location['id']}")
        location_hv = location_node_encoder.encode_node(
            location,
            location_semantic_hv,
        )
        location_hvs[location["id"]] = location_hv
        location_ids.append(location["id"])
        location_vectors.append(location_hv)

    predicate_hv = encoder.predicate_hv(PREDICATE)
    relationship_ids = []
    relationship_vectors = []
    for relationship in relationships.iter_rows(named=True):
        triple_hv = encoder.encode_triple(
            person_hvs[relationship["person_id"]],
            predicate_hv,
            location_hvs[relationship["location_id"]],
            subject_context=f"Person:{relationship['person_id']}:properties",
            object_context=f"Location:{relationship['location_id']}:full-node",
        )
        relationship_ids.append(relationship["id"])
        relationship_vectors.append(triple_hv)

    return EncodedRows(
        EncodedTable(person_ids, torch.stack(person_vectors)),
        EncodedTable(location_ids, torch.stack(location_vectors)),
        EncodedTable(relationship_ids, torch.stack(relationship_vectors)),
    )


def merge_hv_column(db_uri: Path, encoded: EncodedRows) -> None:
    """Join the new HDC vector column onto existing graph rows by `id`.

    Uses Lance's column merge rather than a row upsert: `merge_insert` cannot
    round-trip a blob column (it reads existing values back as descriptors), so
    we only ever bring in the `id` key plus the freshly computed vectors and let
    Lance graft them on. Existing columns, including the lazy image blob, are
    untouched. Any prior HDC columns are dropped first so re-encoding is
    idempotent.
    """
    for table_name, table in [
        ("Person", encoded.persons),
        ("Location", encoded.locations),
        (PREDICATE, encoded.relationships),
    ]:
        path = dataset_path(db_uri, table_name)
        ds = lance.dataset(path)
        if "hv" in ds.schema.names:
            ds.drop_columns(["hv"])
            ds = lance.dataset(path)
        ds.merge(
            hv_merge_table(
                table.ids,
                table.hvs,
                id_type=ds.schema.field("id").type,
            ),
            left_on="id",
            right_on="id",
        )


def hv_merge_table(
    ids: list[str],
    hvs: torch.Tensor,
    *,
    id_type: pa.DataType = pa.large_string(),
) -> pa.Table:
    """Build the `id` + hypervector Arrow table Lance grafts onto a graph table.

    This path keeps whole batches in tensor/NumPy/Arrow buffers; it never
    materializes 10,000 coordinates per row as Python lists. TorchHD performs
    all algebra in float32, and the flattened buffer is downcast to float16
    only at this final storage boundary.
    """
    if hvs.shape != (len(ids), DIMENSIONS):
        raise ValueError(
            f"Invalid hv tensor shape {tuple(hvs.shape)} for {len(ids)} rows"
        )
    flat = (
        hvs.detach()
        .to(dtype=torch.float32, device="cpu")
        .numpy()
        .astype("float16", copy=False)
        .reshape(-1)
    )
    hv_array = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.float16()),
        DIMENSIONS,
    )
    return pa.Table.from_arrays(
        [pa.array(ids, type=id_type), hv_array],
        names=["id", "hv"],
    )


def load_or_create_projector(
    db_uri: Path,
    embeddings: torch.Tensor,
    embedder: OllamaEmbedder,
    model_digest: str,
    projection_seed: int,
    semantic_weight: int,
) -> tuple[RandomHyperplaneProjector, SemanticHDCMetadata]:
    metadata_path = db_uri / SEMANTIC_HDC_METADATA_FILENAME
    projection_path = db_uri / SEMANTIC_PROJECTION_FILENAME
    artifacts_exist = (metadata_path.exists(), projection_path.exists())
    if artifacts_exist == (True, False) or artifacts_exist == (False, True):
        raise RuntimeError(
            "Semantic HDC metadata and projection must either both exist or both be absent"
        )

    if metadata_path.exists():
        metadata = SemanticHDCMetadata.read(metadata_path)
        if projection_seed != metadata.projection_seed:
            raise ValueError(
                "Projection seed changed; rebuild the semantic HDC artifacts"
            )
        metadata.validate_runtime(
            model=embedder.canonical_model,
            model_digest=model_digest,
            embedding_dimensions=int(embeddings.shape[1]),
        )
        projector = RandomHyperplaneProjector.load(
            db_uri / metadata.projection_file,
            seed=metadata.projection_seed,
            expected_sha256=metadata.projection_sha256,
        )
        if projector.output_dimensions != metadata.hdc_dimensions:
            raise ValueError("Projection output dimension does not match metadata")
        return projector, replace(metadata, semantic_weight=semantic_weight)

    projector = RandomHyperplaneProjector.create(
        input_dimensions=int(embeddings.shape[1]),
        output_dimensions=DIMENSIONS,
        seed=projection_seed,
    )
    projection_sha256 = projector.save(projection_path)
    metadata = SemanticHDCMetadata(
        model=embedder.canonical_model,
        model_digest=model_digest,
        embedding_dimensions=projector.input_dimensions,
        hdc_dimensions=projector.output_dimensions,
        projection_seed=projection_seed,
        projection_type=PROJECTION_TYPE,
        projection_file=SEMANTIC_PROJECTION_FILENAME,
        projection_sha256=projection_sha256,
        semantic_weight=semantic_weight,
    )
    return projector, metadata


def encode_hdc(
    db_uri: Path = DEFAULT_DB_URI,
    raw_data_dir: Path = RAW_DATA_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    projection_seed: int = SEED,
    semantic_weight: int = DEFAULT_SEMANTIC_WEIGHT,
) -> None:
    """Add HDC vector columns to graph tables in the shared LanceDB dataset.

    Run graph_ingest.py first. This step intentionally augments those same
    tables, demonstrating LanceDB's ability to evolve the schema with new
    feature columns.
    """
    persons = read_table(db_uri, "Person")
    locations = read_table(db_uri, "Location")
    relationships = read_table(db_uri, PREDICATE)
    evidence_rows = location_evidence_records(raw_data_dir)
    validate_location_evidence(locations, evidence_rows)
    location_evidence = prepare_location_evidence(evidence_rows)

    documents = build_location_documents(location_evidence)
    location_ids = sorted(documents)
    embedder = OllamaEmbedder(model=embedding_model, host=ollama_host)
    model_digest = embedder.model_digest()
    embeddings = embedder.embed_documents(
        [documents[location_id] for location_id in location_ids]
    )
    projector, metadata = load_or_create_projector(
        db_uri,
        embeddings,
        embedder,
        model_digest,
        projection_seed,
        semantic_weight,
    )
    semantic_hvs = projector.project(embeddings)
    location_semantic_hvs = dict(zip(location_ids, semantic_hvs))

    encoder = TorchHDEncoder()
    location_node_encoder = LocationNodeEncoder(
        encoder,
        semantic_weight=semantic_weight,
    )
    encoded = encode_rows(
        persons,
        locations,
        relationships,
        location_semantic_hvs,
        encoder,
        location_node_encoder,
    )
    merge_hv_column(db_uri, encoded)
    metadata.write(db_uri / SEMANTIC_HDC_METADATA_FILENAME)
