# Multimodal Knowledge Graphs with HDC + lance-graph

A small, self-contained demo that combines two ways of looking at the same data: hyperdimensional space (10K dimensions) and property graphs. [LanceDB](https://lancedb.com) is used as the storage, providing two forms of retrieval over the data:

- **Associative search** via **Hyperdimensional Computing (HDC)**: a local text-embedding model
  supplies semantic similarity, then a random projection carries that geometry into 10,000
  dimensions. This answers paraphrased questions such as *"mountainous cities near the pacific
  ocean"* without an exact-term vocabulary.
- **Property-graph traversal** via **[lance-graph](https://github.com/lancedb/lance-graph)**:
  exact, schema-valid Cypher over the same tables.

Both views live in **one [Lance](https://lance.org)** dataset. Lance is a multimodal lakehouse format that's well suited for HDC.
Graph facts, hypervectors, and native storage of multimodal assets (images, video, sensor traces, and more) are all columns of the same table, indexed and versioned together, with no need to manage multiple systems to keep them in sync. LanceDB is the data management platform and lakehouse built on top of the Lance format.

The core idea in this demo is this: we let HDC propose **could-be-true** candidates from fuzzy intent-style queries, and let the graph confirm **what is known to be true**, based on the facts it stores.

A knowledge graph and a hyperdimensional space are just two
representations of the same data in two different topological spaces: one discrete and exact, one
continuous and fuzzy.

> [!NOTE]
> This demo is deliberately tiny (4 people, 3 cities). The multimodal "vibe" features in
> [data/features/location_vibes.csv](data/features/location_vibes.csv) are hand-authored
> stand-ins shaped exactly like the captions a vision model (CLIP / a vision-LLM) would emit. The
> semantic regression set is likewise a small demo sanity check, not a general retrieval benchmark.

## What is HDC?

**Hyperdimensional Computing (HDC)** represents every concept (a person, a city, a feature like
"mountains") as a single hypervector (here, 10,000 numbers). Structural concepts such as graph IDs,
property keys, and predicates use deterministic hash-derived random hypervectors, so they remain
distinct and require no vocabulary file. Semantic location descriptions use a text embedding plus
the random up-projection described below. You then build representations with
three basic operations, the algebra of high-dimensional random hypervectors introduced by Pentti Kanerva
([Kanerva, 2009](https://doi.org/10.1007/s12559-009-9009-8)):
- **Bundling** (add hypervectors together to form a set or "bag" of things, where the
result stays *similar* to each ingredient)
- **Binding** (multiply hypervectors to tie a role to a
value, where the result is *dissimilar* to its parts but can be cleanly undone later).
- **Permuting** (shuffle a hypervector's coordinates in a fixed, reversible pattern, usually a cyclic
shift, so the result is *dissimilar* to the original but can be undone; this is how order or position
gets encoded, e.g. telling the first item in a sequence apart from the second)

This demo uses only bundling and binding; permutation is included here for completeness, since it
rounds out the standard HDC toolkit even though the person/location graph has no ordered sequences to encode.

These three mathematical operations are enough to encode a whole record as one hypervector and to ask fuzzy, compositional questions of it
by comparing hypervectors with a similarity score. Because the space is continuous, answers *degrade
gracefully*: a city that matches most of a query scores high, one that matches only part of it
scores lower: no exact keyword or column ever has to match. Cosine scores are ranking signals, not
probabilities, and their absolute baseline depends on the embedding model and corpus. That soft,
similarity-based matching is what complements the exact, schema-bound graph
traversal in the rest of this demo.

## The data

The graph follows a compact pattern:

```cypher
(:Person)-[:VISITED]->(:Location)
```

Four people across three cities: Robby Jo and Maya Chen in Seattle, Andre Brooks in New York, and
Elena Park in Salt Lake City. The source of truth is plain CSV under [data/](data/), loaded with
Polars:

| File | Role |
|------|------|
| [data/nodes/person.csv](data/nodes/person.csv) | Person nodes |
| [data/nodes/location.csv](data/nodes/location.csv) | Location nodes (with `timezone`, `region`, lat/lon buckets, `image_path`) |
| [data/relationships/visited.csv](data/relationships/visited.csv) | `VISITED` edges |
| [data/features/location_vibes.csv](data/features/location_vibes.csv) | Derived multimodal features (`mountains`, `pacific_coast`, `waterfront`, …) with VLM-style `evidence` captions |
| [img/](img/) | Skyline images (`seattle.jpg`, `nyc.jpg`, `salt-lake-city.jpg`) |

Note the deliberate mismatch that motivates the whole demo: `timezone` (`pacific`/`eastern`/`mountain`)
is a real graph column, but `pacific_coast` is **not**: it exists only as a derived vibe feature.
Exact Cypher can't match a concept that was never a column; HDC can.

## How it works

The build runs in two stages, both writing to the same dataset:

1. **`graph_ingest.py`** ingests the raw CSVs into a LanceDB dataset (`person-location/`) as three
   tables (`Person`, `Location`, `VISITED`). The graph facts *and* the raw skyline image bytes
   land here together.
2. **`hdc_encode.py`** asks a locally running Ollama `nomic-embed-text` model to embed one
   evidence document per location, projects those embeddings into bipolar 10,000-dimensional HDC
   space, and adds one full-node HDC column to each of the same tables in place.

The payoff is that every row carries graph facts, a native multimodal asset, and hypervectors side
by side (this is what makes it a multimodal knowledge graph, not a graph plus three bolt-on stores).
Here is every column of every table:

**`Person`**

| Column | Type | Holds |
|--------|------|-------|
| `id` | string | Stable node id (`person-1`, …) |
| `name` | string | Person name |
| `kind` | string | Node label (`Person`) |
| `role` | string | Occupation (`engineer`, `designer`, …) |
| `hv` | vector[10000] | Property-bag hypervector |

**`Location`**

| Column | Type | Holds |
|--------|------|-------|
| `id` | string | Stable node id (`location-seattle`, …) |
| `name` | string | City name |
| `kind` | string | Node label (`Location`) |
| `country` | string | Country code |
| `region` | string | Region (`pacific_northwest`, …) |
| `timezone` | string | `pacific` / `eastern` / `mountain` |
| `lat_bucket` | string | Coarse latitude bucket |
| `lon_bucket` | string | Coarse longitude bucket |
| `description` | string | Free-text city description |
| `image_path` | string | Human-readable pointer to the skyline photo |
| `image` | **binary (blob)** | **Raw JPEG bytes of the skyline photo, stored natively as a lazy blob** |
| `hv` | vector[10000] | Full location hypervector: structural property sum plus weighted semantic evidence |

**`VISITED`**

| Column | Type | Holds |
|--------|------|-------|
| `id` | string | Stable edge id |
| `person_id` | string | Source `Person.id` |
| `location_id` | string | Target `Location.id` |
| `predicate` | string | Edge type (`VISITED`) |
| `hv` | vector[10000] | S-P-O binding `hv(person) * hv(VISITED) * hv(location)` |

`LocationNodeEncoder` makes the location-specific lifecycle explicit. For a row such as Seattle,
it first encodes every non-vector, non-blob property as a structural property bag. It then combines
that raw sum with an 8× weighted semantic term:

```text
Location.hv = structural_sum + (8 × semantic_hv)

complete Seattle row ─→ structural sum ─────────────┐
                                                     ├─ add ─→ Location.hv
vibe evidence ─→ embedding ─→ projected semantic hv ─┘  weight = 8
```

The structural and semantic vectors are transient encoding components, not stored columns.
`Location.hv` is both the full node representation used for graph composition and the vector
searched by free-text semantic queries. A Location row contributes ten structural associations,
so an unweighted semantic vector would be diluted by that larger property bundle. Weighting the
semantic term by 8 keeps its geometry visible to natural-language cosine search while retaining
the structural signal in the same vector. The weight is recorded in `semantic_hdc.json` and should
be tuned for other schemas.

`Location.image` holds the actual bytes of `img/seattle.jpg` and its siblings (JPEG, ~9 to 11 KB
each), not a link to them. Graph facts, the image asset, and the hypervectors are columns of one
LanceDB row: indexed and versioned together, with nothing to keep in sync across systems.

The column is written with Lance's blob encoding (`lance-encoding:blob`), so the bytes are **lazy**:
ordinary scans and Cypher queries return a small `{position, size}` descriptor, never the image
itself. The bytes are read only when something explicitly asks for them via `take_blobs` (see the
`/image/{label}/{id}` endpoint in [src/graph/api.py](src/graph/api.py)). A query that never wants a
picture never pays to read one, which is what makes storing large assets inline practical.

At query time ([src/hdc/retrieve.py](src/hdc/retrieve.py)), the raw natural-language query goes through the same embedding
model and projection as the stored location documents. The resulting query hypervector ranks
`Location.hv` by cosine similarity. There is no supported-phrase list, query-to-feature
mapper, token lookup, or `vocabulary.json`. `lance-graph` ([src/graph/retrieve.py](src/graph/retrieve.py)) then traverses from
each candidate location to the people connected by real `VISITED` edges (the validation step).

### From a text embedding to HDC space

The embedding model returns one dense vector per piece of text. The locally tested
`nomic-embed-text:latest` build returns 768 numbers. Those 768 values already contain the semantic
information: phrases such as "mountainous" and "mountains," or "Pacific Ocean" and "pacific
coast," land near one another even though their words are not identical.

The up-projection does **not** invent more semantic information. It spreads the existing geometry
across many HDC coordinates:

1. Create one fixed random matrix `R` with shape `768 × 10,000`, containing only `-1` and `+1`.
2. Multiply the normalized embedding `x` by that matrix: `z = xR`.
3. Keep only the sign of each result: positive becomes `+1`, negative becomes `-1`.

Each output coordinate is therefore a random hyperplane vote over all 768 embedding values. One
vote is noisy; 10,000 independent votes concentrate around a stable answer. Semantically close
embeddings tend to fall on the same side of most hyperplanes and produce similar bipolar
hypervectors. Put simply: similar embeddings agree on more of these 10,000 yes/no votes, while
unrelated embeddings agree mostly by chance. The representation changes, but the useful notion of
which texts are closer is largely retained.

The exact same model digest, normalization, and persisted matrix are used for documents and
queries:

```text
location evidence ─→ search_document embedding ─→ sign(xR) ─→ weighted into Location.hv
arbitrary query   ─→ search_query embedding    ─→ sign(xR) ─→ cosine search over Location.hv
```

The 7.3 MB `int8` matrix is a coordinate-system artifact, not a codebook: it contains no words,
features, or lookup entries. Metadata records its checksum, seed, dimensions, template version,
and Ollama model digest so query-time code fails clearly instead of silently mixing incompatible
spaces.

### Vocabulary-free encoding

For historical context, the prototype used `vocabulary.json` plus a hard-coded query-to-feature
translator. The current implementation has neither. Instead:

- Structural graph symbols are generated independently from stable hashes, eliminating the global
  vocabulary while preserving nearly orthogonal identities and exact MAP algebra.
- Fuzzy location text and raw queries are embedded by the same semantic model and projected through
  the same matrix, allowing unseen wording and paraphrases to meet in HDC space.

The graph remains responsible for exact facts, while embedding-projected HDC is the semantically
friendly candidate generator.

### HDC primitives (via [TorchHD](https://github.com/hyperdimensional-computing/torchhd))

Defined in [src/hdc/core.py](src/hdc/core.py):

- **Structural hypervector**: a 10,000-dimensional bipolar MAP vector derived independently from a
  stable hash of an ID, key, value, or predicate. Any unseen symbol can be encoded immediately;
  adding one never changes existing symbols.
- **Semantic hypervector**: a normalized text embedding transformed by the shared random-hyperplane
  projection above.
- **Binding** (`multibind`): associates hypervectors; the result is *dissimilar* to its parts and is
  reversible. Used to encode `subject * predicate * object`.
- **Bundling** (`bundle` / `torchhd.multiset`): superposition; the result stays *similar* to each
  ingredient. Used to accumulate a node's property/feature bag.

## Hypervector representation lifecycle

Every hypervector here uses **MAP**. MAP ("Multiply-Add-Permute") is the vector-symbolic
model TorchHD uses by default (each atomic token is a 10,000-dim hypervector of `±1`): **bundling** is
element-wise *addition*, and **binding** is element-wise *multiplication*. 

Mathematically, both binding and bundling are **exactly invertible** operations, but in practice (when working with TorchHD), we have to understand when to store normalized vs. raw hypervectors so that the original hypervectors are recoverable after running numerical operations on them.

- **Binding is exactly invertible: but only when every factor is `±1`.** Because `(+1)² = (-1)² =
  1`, each factor is its own inverse, so `a * b * c` multiplied by the known `a` and `b` recovers
  `c` exactly. Feed in a factor with any other value and that guarantee is gone.
- **Bundling is exactly invertible: but only if you keep the sum intact.** The bundle of A, B, C
  is the literal element-wise sum, so its coordinates are small integers (…, `-2`, `0`, `3`, …)
  that record *how many* ingredients (and *how strongly*) voted at each position. Add or subtract
  a member and you land exactly on the smaller bundle. Collapse that sum back down to `±1` and the
  counts are lost for good.

The un-collapsed, integer-valued hypervectors are **unnormalized**, and the sign-only `±1` version is
their **normalized (bipolar)** form. Binding wants bipolar factors; structural property bundles
keep their unnormalized sums. A projected semantic location document is already one bipolar
factor—it is not a weighted bundle of exact vocabulary terms.

### Storage: what each LanceDB table holds

We create multiple LanceDB tables as follows:

- **`Person.hv` stores an unnormalized property sum.** It is a full-precision bundle of
  hash-derived role/value associations. Keeping the integer coordinates preserves exact additive
  insert/remove, which is why `bundle()` never normalizes internally.
- **`Location.hv` is the only stored Location vector.** It keeps the raw structural property sum
  and an 8× weighted bipolar semantic term. Updating source evidence rebuilds this full vector; no
  separate structural or semantic component column is retained.
- **`VISITED.hv` stores one bipolar relationship product.** Before a
  node sum enters an S-P-O binding, `normalize_for_binding()` collapses it to `{-1, +1}` so the
  multiply stays self-inverse and a known subject + predicate recover the encoded object exactly.
- **Zero coordinates are broken deterministically.** An even-sized unnormalized sum can land on exactly `0`,
  which has no sign to keep; a stable context (e.g. `Location:seattle`) seeds a random `±1` tie
  hypervector, avoiding a global `0 -> -1` bias while keeping rebuilds reproducible.

Normalize a composite only at the node → relationship-binding boundary. The full, unnormalized
`Location.hv` remains suitable for cosine retrieval and preserves the weighted superposition.

## Setup

Requires **Python 3.13+**, [uv](https://docs.astral.sh/uv/), and a local
[Ollama](https://ollama.com) server. Download `nomic-embed-text` into Ollama's local model store
using the [Ollama CLI](https://docs.ollama.com/cli), then install the Python dependencies:

```bash
# Start Ollama first if your installation does not run it as a background service.
ollama serve

# In another terminal, pull and verify the local embedding model.
ollama pull nomic-embed-text
ollama ls

uv sync
```

Direct dependencies include the official `ollama` Python client, `lancedb`, `lance-graph`,
`polars`, `pyarrow`, `numpy`, `torch`, `torch-hd`, plus `fastapi` / `uvicorn` for the visualizer
backend. The client talks to the local Ollama server, so the demo adds no hosted embedding service
or API key.

You can install additional dependencies as needed using the `uv add` command.

### Code layout

The Python implementation is organized by responsibility:

```text
src/
├── graph/        # source data, Lance ingestion, Cypher traversal, visualizer API
├── hdc/          # MAP algebra, semantic projection, full-node encoding, retrieval
├── evaluation/   # semantic ranking and VISITED path regression logic
└── *.py          # thin executable entry points
```

This keeps the commands below stable while the reusable code lives in focused modules.

Run the pytest suite with:

```bash
uv run pytest
```

## Running the demo

The end-to-end path builds the dataset and runs the combined fuzzy + graph query:

```bash
uv run python src/run_person_location_demo.py
```

By default it asks *"mountainous cities near the pacific ocean"*. Neither `mountainous` nor
`pacific ocean` is an exact feature label:

```text
Question: mountainous cities near the pacific ocean

LanceDB HDC matches, expanded through lance-graph:
  - Maya Chen -> Seattle (pacific, score=0.507, features=mountains, pacific_coast, scenic_urban)
  - Robby Jo -> Seattle (pacific, score=0.507, features=mountains, pacific_coast, scenic_urban)
  - Elena Park -> Salt Lake City (mountain, score=0.434, features=mountains, nature_access, mountain_west)
  - Andre Brooks -> New York (eastern, score=0.384, features=dense_skyline, urban_energy, concrete_jungle)

All lance-graph validation paths:
  - Andre Brooks -> New York (eastern)
  - Elena Park -> Salt Lake City (mountain)
  - Maya Chen -> Seattle (pacific)
  - Robby Jo -> Seattle (pacific)
```

Seattle ranks first because the embedding connects the paraphrases to Seattle's evidence about
mountains, Puget Sound, and the Pacific Northwest. Salt Lake City remains a sensible runner-up
because it strongly matches the mountainous part.

There is no default score threshold. Semantic cosine scores are not probabilities, so retrieval is
top-k by default. `--min-score` remains available for thresholds calibrated against a
representative query set.

Pass your own query with `--query`:

```bash
uv run python src/run_person_location_demo.py --query "concrete jungle"
```

### Running the stages separately

**1. Build the graph tables, then add HDC columns.** Run these once before any query below:

```bash
uv run python src/graph_ingest.py
uv run python src/hdc_encode.py
```

```text
Wrote shared LanceDB graph tables: .../person-location
Added HDC columns to shared LanceDB dataset: .../person-location
Wrote semantic HDC metadata and projection: .../person-location/semantic_hdc.json
```

**2. Exact graph queries (lance-graph / Cypher).** These match on real schema columns only:

```bash
uv run python src/graph_retrieve.py --city Seattle
```

```text
{'person': 'Maya Chen', 'city': 'Seattle'}
{'person': 'Robby Jo', 'city': 'Seattle'}
```

```bash
uv run python src/graph_retrieve.py --timezone pacific
```

```text
{'person': 'Maya Chen', 'city': 'Seattle', 'timezone': 'pacific'}
{'person': 'Robby Jo', 'city': 'Seattle', 'timezone': 'pacific'}
```

```bash
uv run python src/graph_retrieve.py --query "MATCH (p:Person)-[:VISITED]->(l:Location) RETURN p.name AS person, l.name AS city ORDER BY city, person"
```

```text
{'person': 'Andre Brooks', 'city': 'New York'}
{'person': 'Elena Park', 'city': 'Salt Lake City'}
{'person': 'Maya Chen', 'city': 'Seattle'}
{'person': 'Robby Jo', 'city': 'Seattle'}
```

**3. Fuzzy full-node HDC + graph-path queries.** These embed arbitrary text into a semantic query
hypervector, rank the weighted full-node `Location.hv` values by cosine similarity, then expand
candidate cities to people through exact `VISITED` edges:

```bash
uv run python src/hdc_retrieve.py \
  --query "mountainous cities near the pacific ocean" \
  --limit 2
```

```text
{'person': 'Maya Chen', 'city': 'Seattle', ..., 'score': 0.507}
{'person': 'Robby Jo', 'city': 'Seattle', ..., 'score': 0.507}
{'person': 'Elena Park', 'city': 'Salt Lake City', ..., 'score': 0.434}
```

```bash
uv run python src/hdc_retrieve.py \
  --query "arid basin city surrounded by mountain ranges"
```

```text
{'person': 'Elena Park', 'city': 'Salt Lake City', ..., 'score': 0.446}
{'person': 'Maya Chen', 'city': 'Seattle', ..., 'score': 0.406}
{'person': 'Robby Jo', 'city': 'Seattle', ..., 'score': 0.406}
```

The wording contains neither the stored `mountains` feature label nor Salt Lake City's literal
description, but the shared embedding space connects the dry inland-basin phrasing to its
evidence.

The query can express both fuzzy location intent and the desired graph path:

```bash
uv run python src/hdc_retrieve.py \
  --query "persons who visited cities on the pacific coast" \
  --limit 1
```

```text
{'person': 'Maya Chen', 'city': 'Seattle', ..., 'score': 0.410}
{'person': 'Robby Jo', 'city': 'Seattle', ..., 'score': 0.410}
```

The semantic HDC stage recognizes Seattle as the top location; the exact graph stage then returns
both people connected to Seattle by `VISITED`. The embedding does not guess the visitors.

```bash
uv run python src/hdc_retrieve.py --query "concrete jungle"
```

```text
{'person': 'Andre Brooks', 'city': 'New York', ..., 'score': 0.396}
```

The runner rebuilds `person-location/`, including the semantic metadata and 7.3 MB projection
artifact, from the current CSV rows each time. Structural symbols are hash-derived on demand; no
vocabulary is built or stored.

To grow the demo, add rows to the CSVs under [data/](data/), then rerun `graph_ingest.py` followed by
`hdc_encode.py` (the HDC step assumes the graph tables already exist).

### Semantic and graph-path regression evaluation

[data/evaluation/semantic_queries.csv](data/evaluation/semantic_queries.csv) is the hand-authored
regression set for the current one-vector approach. Its nine queries are intentionally
discriminative for this three-city corpus and should remain 9/9. It is not a held-out benchmark
and is too small to claim general retrieval quality; its job is to catch regressions in paraphrase
handling, projection, ranking, and model/template compatibility.

[data/evaluation/semantic_path_queries.csv](data/evaluation/semantic_path_queries.csv) adds three
end-to-end cases. Each first ranks a city through `Location.hv`, then asserts the exact people
reached through stored `VISITED` edges. Run both sets against an encoded dataset:

```bash
uv run python src/evaluate_semantic_hdc.py
```

Current local results with `nomic-embed-text:latest`, model digest recorded in
`semantic_hdc.json`, and projection seed 13:

| Query | Expected / top result | Score | Runner-up | Margin |
|---|---|---:|---|---:|
| mountainous cities near the pacific ocean | Seattle | 0.507 | Salt Lake City (0.434) | 0.073 |
| waterfront skyline backed by rugged peaks | Seattle | 0.436 | New York (0.392) | 0.044 |
| Pacific Northwest city with easy access to nature and water | Seattle | 0.493 | Salt Lake City (0.441) | 0.052 |
| arid basin city surrounded by mountain ranges | Salt Lake City | 0.446 | Seattle (0.406) | 0.040 |
| an inland alpine city close to skiing and mountain trails | Salt Lake City | 0.468 | Seattle (0.446) | 0.022 |
| mountain west city beside a huge salty lake | Salt Lake City | 0.434 | Seattle (0.392) | 0.042 |
| concrete jungle with dense skyscrapers | New York | 0.534 | Seattle (0.366) | 0.168 |
| dense built environment with iconic towers | New York | 0.509 | Seattle (0.375) | 0.134 |
| urban energy among crowded high rises | New York | 0.415 | Seattle (0.345) | 0.070 |

| VISITED query | Top city | Score | Exact visitors |
|---|---|---:|---|
| persons who visited cities on the pacific coast | Seattle | 0.410 | Maya Chen, Robby Jo |
| who visited a dense concrete jungle of skyscrapers | New York | 0.464 | Andre Brooks |
| who traveled to an inland alpine city near ski trails | Salt Lake City | 0.443 | Elena Park |

The Pacific-coast query correctly returns both Maya and Robby: both have an exact `VISITED` edge
to Seattle. The semantic stage chooses Seattle; the graph stage—not the embedding—determines its
visitors.

```text
Semantic top-1 accuracy: 9/9 (100.0%)
Semantic mean reciprocal rank: 1.000
VISITED path checks: 3/3 (100.0%)
```

These are scores read from the persisted float16 `Location.hv` column, not from the transient
semantic component. Margins remain useful diagnostics, but top-1 accuracy on nine regression
queries should not be mistaken for calibrated confidence or a general benchmark.

## Visualizing the graph

A two-view React + FastAPI visualizer lives in [viz/](viz/):

- **Schema view**: the meta-graph (labels, properties, relationships) derived from the graph config
  and each table's Arrow schema. The hypervector column (`hv`) and blob asset (`image`) are badged,
  never drawn.
- **Instance view**: a query *builder*. Pick a relationship, per-label display/tooltip columns, and
  structured filters. The backend constructs Cypher from your selections and returns a node-link
  graph; the generated Cypher is shown read-only. Click a `Location` node and its skyline image is
  fetched on demand from the lazy blob column (nothing else loads image bytes).

The Cypher engine (`lance_graph.CypherEngine`) is Python-only, so the browser never touches it: a
FastAPI backend serves JSON and the React frontend draws it.

Run the two processes (build the dataset first if you haven't):

```bash
# 1. Backend (port 8000): uses build_engine() from src/graph/retrieve.py
uv run uvicorn graph_api:app --app-dir src --reload

# 2. Frontend (must be port 5173; the backend CORS allowlist pins it)
cd viz && npm install && npm run dev
```

Then open **http://localhost:5173**. The preview will showcase a property graph visualization as follows:

![](./img/graph-preview.png)

To point the whole stack at a different dataset, edit `GRAPH_SCHEMA` in
[src/graph/retrieve.py](src/graph/retrieve.py): both the engine and the visualizer follow it.
See [viz/README.md](viz/README.md) for the API contract.

## Cost caveats

- **Embedding compute**: document encoding and every query call the local Ollama model. Source
  documents are embedded in one batch; production ingestion should add a content-addressed cache
  keyed by text, model digest, and template version.
- **Projection compute**: the fixed `768 × 10,000` Rademacher matrix is about 7.3 MB as `int8`.
  Multiplication runs in float32 and is batchable. The expansion adds redundancy and an
  HDC-compatible representation, not new semantic information.
- **Reproducibility**: changing the model digest, document template, input dimension, or projection
  artifact changes the coordinate system. The stored metadata detects these changes and requires a
  rebuild.
- **Storage**: encoded hypervectors are computed in float32, then stored in Lance as float16—a
  deliberate storage/precision tradeoff that is lossless for the demo's small integer structural
  sums and bipolar semantic vectors. One stored hypervector is 10,000 dims × 2 bytes = **20 KB**,
  with exactly one `hv` per Person, Location, or relationship row:

  | Scale | Hypervectors stored | Approx. raw size (float16, pre-indexing) |
  |-------|----------------|------------------------------------------|
  | This demo | a handful | kilobytes |
  | 1M locations × `hv` | 1M hypervectors | ~20 GB |
  | 1M edges × `hv` | 1M hypervectors | ~20 GB |

Compression / quantization (binary/bipolar packing, dimensionality choices, learned compression) and ANN indexing of the `hv` columns are real levers a production system would need, which [LanceDB](https://lancedb.com) is well-suited for.
See [COMPRESSION_TRICKS.md](COMPRESSION_TRICKS.md) for concrete options, tradeoffs, and measured binary-HDC results from this demo.
