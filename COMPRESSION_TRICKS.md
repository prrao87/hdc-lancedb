# HDC storage and compression tricks

This repository uses 10,000-dimensional hypervectors. That is intentionally
large enough to make the HDC behavior easy to demonstrate, but it also makes
storage a first-class design concern. In this document, **hypervector** means an
HDC representation such as the 10,000-dimensional MAP tensors in this repo.
**Vector** is reserved for general-purpose numerical vectors, conventional
text/image embeddings, and LanceDB API terms such as "vector search."

The raw cost per hypervector is:

| Representation | Bytes per dimension | Approx. bytes per hypervector | Approx. size for 1M hypervectors |
|---|---:|---:|---:|
| `float32` | 4 | 40,000 | 40 GB |
| `float16` | 2 | 20,000 | 20 GB |
| `int8` | 1 | 10,000 | 10 GB |
| Packed binary | 1 bit | 1,250 | 1.25 GB |

These are decimal, pre-indexing estimates and exclude row metadata, index
structures, and retained dataset versions.

## Start by storing fewer hypervectors

The best compression is not materializing a hypervector that no query uses.

The current demo stores `Person.hv`, `Location.hv`, `Location.vibe_hv`,
`VISITED.hv`, and `VISITED.vibe_hv`, but its online retrieval path searches only
`Location.vibe_hv`. The other hypervectors illustrate HDC composition and
reversible relationships; a production system should decide which of them
deserve online materialization.

Useful patterns include:

- Store only hypervectors used by an online query or index.
- Regenerate deterministic atomic hypervectors from the vocabulary, seed,
  dimensions, and model instead of persisting the atomic embedding matrix.
- Recompute cheap derived hypervectors from source facts when they are rarely read.
- Avoid copying a location hypervector onto every incident edge. Retrieve locations
  first, then traverse graph edges by ID, as this demo does.
- Keep cold or experimental hypervector representations in a separate table
  keyed by stable entity ID.
- Project out existing hypervectors and blobs during re-encoding so old large columns
  are never read just to be discarded.

## Use narrower floating-point storage

This repository computes TorchHD algebra in `float32` and stores finished Lance
columns as `float16`. This halves raw storage while keeping TorchHD on one of its
officially supported MAP dtypes.

The current MAP bundles contain small integer-valued coordinates. `float16`
represents every integer through 2,048 exactly, so values such as `-30`, `-4`,
and `2` survive the storage cast without loss. Freshly regenerated `float32`
hypervectors and the persisted `float16` hypervectors produce the same Seattle
and Salt Lake City cosine scores in this dataset.

The safe boundary is:

```text
TorchHD binding/bundling: float32
             ↓
Arrow/Lance storage:      float16
             ↓
Reload for TorchHD:       explicitly upcast to float32
```

Do not rely on casting a `MAPTensor` to `float16` and continuing TorchHD
operations. PyTorch supports half precision, but TorchHD 5.8.4 does not declare
`float16` as a supported MAP dtype. The cast also becomes lossy once bundle
counts grow beyond the exact-integer range of half precision.

## Binary HDC and Hamming search

A binary-first design is practical for this workload and offers the largest
simple reduction in raw hypervector storage.

### Representation

Map bipolar MAP coordinates to bits:

```text
-1 → 0
+1 → 1
```

Then pack eight logical dimensions into every byte. With 10,000 dimensions:

```text
10,000 / 8 = 1,250 uint8 values per hypervector
```

The Lance field is therefore a fixed-size list of 1,250 bytes, not an integer
array of length 10,000:

```python
import numpy as np
import pyarrow as pa

bits = bipolar_hv.cpu().numpy() > 0
packed = np.packbits(bits).astype(np.uint8)
binary_type = pa.list_(pa.uint8(), 10_000 // 8)
```

[LanceDB calls this representation a packed binary
vector](https://docs.lancedb.com/search/vector-search) whose logical dimension
is divisible by eight. Binary indexing uses Hamming distance with `IVF_FLAT`:

```python
from lancedb.index import IvfFlat

table.create_index(
    "vibe_hv_binary",
    config=IvfFlat(distance_type="hamming"),
)
```

See the [LanceDB vector-index guide](https://docs.lancedb.com/indexing/vector-index)
for the current API and binary-index constraints.

### Operations

| HDC operation | Bipolar MAP | Packed binary equivalent |
|---|---|---|
| Binding | Element-wise multiplication | XOR |
| Unbinding | Element-wise multiplication | XOR |
| Permutation | Coordinate rotation | Bit rotation/permutation |
| Similarity | Cosine on bipolar hypervectors | Hamming distance |
| Bundling | Add vote counts | Count votes, then majority threshold |

For bipolar hypervectors of dimension \(D\), cosine similarity and Hamming
distance \(H\) are exactly related:

\[
\operatorname{cosine}(A,B) = 1 - \frac{2H(A,B)}{D}
\]

An application can convert LanceDB's Hamming distance into a centered score:

```python
score = 1.0 - 2.0 * hamming_distance / dimensions
```

Unrelated binary hypervectors then score near zero, just as unrelated bipolar
hypervectors do under cosine similarity.

### Bundling is the tradeoff

Binary hypervectors cannot directly retain a raw bundle such as:

```text
[-30, -26, -4, 0, 8, 30, ...]
```

Construction requires a temporary wider accumulator:

```text
component hypervectors
        ↓ add votes in int16/int32/float32
raw bundle counts
        ↓ majority threshold with deterministic zero ties
packed binary bundle
```

This discards victory margin: `+2` and `+30` both become `+1`. Feature
repetition still influences which sign wins, but the persisted result no longer
records how decisively it won. Consequences include:

- Exact additive removal from the stored bundle is no longer possible.
- Confidence-sensitive cosine scoring is replaced by majority-sign agreement.
- Close candidates can reorder, and similarity thresholds need recalibration.
- Updates require source evidence, retained counters, or a full bundle rebuild.

This repository retains the source facts and rebuilds HDC columns, so discarding
the counters may be an acceptable production choice. Systems requiring frequent
incremental add/remove operations can keep a compact counter representation in
cold storage while serving packed bits online.

### Zero coordinates need a policy

Even-sized bundles contain ties. The current four-feature query has 3,753 zero
coordinates out of 10,000. A binary hypervector has no zero state.

Mapping every zero to the same bit creates artificial agreement. Prefer a
deterministic pseudo-random tie-breaker derived from stable context:

- Stored entity: stable entity ID and hypervector role.
- Query: stable hash of the normalized query facets.

The existing `normalize_for_binding()` implementation demonstrates this
strategy. A ternary representation could retain zero explicitly, but it would
not use LanceDB's packed-binary Hamming path directly.

### Results on this demo

The current bundles were majority-normalized with context-specific deterministic
zero ties, packed into 1,250 bytes, and searched through LanceDB with Hamming
distance. A binary `IVF_FLAT` index was also created successfully.

| Location | Raw-bundle cosine | Hamming distance | Binary score \(1-2H/D\) |
|---|---:|---:|---:|
| Seattle | 0.741850 | 2,603 | 0.479400 |
| Salt Lake City | 0.443816 | 3,664 | 0.267200 |
| New York | -0.012302 | 5,009 | -0.001800 |

The absolute scores changed, but the ranking and the candidates above the
demo's `0.20` threshold remained the same. Three locations are not enough to
establish production recall, but the result makes binary HDC a credible next
experiment.

## Integer bundle storage

Raw MAP bundles are integer-valued even when represented as floating point. It
is tempting to store them as `int8` or `int16` without majority-thresholding.

This can be exact while every coordinate remains in range:

```text
int8:  -128 through 127
int16: -32,768 through 32,767
```

However:

- Bundle accumulation can silently overflow if performed in the narrow dtype.
- Squaring an `int8` value during a norm calculation can overflow even when the
  original value fits. Promote before dot products, norms, or cosine similarity.
- The LanceDB 0.34.0 numerical vector-search path does not accept a
  `fixed_size_list<int8>` column for cosine, dot, or L2 search. Packed `uint8`
  with Hamming is a separate supported path.
- Per-hypervector scale-and-round quantization needs a stored scale and makes exact
  additive updates harder.

Integer counts can still be useful as a cold, compact source representation or
as temporary bundle accumulators, but they are not currently a drop-in
replacement for the searchable floating-point column.

## Quantized ANN indexes

LanceDB exposes several compressed ANN index representations. These address
index size and query speed without requiring the application to redesign its
HDC algebra:

- **Scalar quantization (`SQ`)**: roughly one code byte per dimension; a useful
  recall/latency/size compromise.
- **Product quantization (`PQ`)**: treats each hypervector as a generic numerical
  vector, partitions it into subvectors, and stores
  compact codebook IDs; usually much smaller but lossy.
- **RaBitQ (`RQ`)**: binary quantization plus correction factors; attractive for
  very high-dimensional hypervectors. The dimensions must be divisible by eight,
  which 10,000 satisfies.
- **Flat indexes**: retain full precision and prioritize recall over size.

See LanceDB's [quantization guide](https://docs.lancedb.com/indexing/quantization)
and [vector-index selection guide](https://docs.lancedb.com/indexing/vector-index).

Quantization can change the candidate set, ordering, and reported distances.
Use `refine_factor` to retrieve extra candidates and rerank them using full
hypervectors when the original column is retained. Also distinguish index storage
from base-table storage: a compressed index does not necessarily eliminate the
raw source column, so measure the size of the complete dataset rather than only
the index codes.

## Two-stage retrieval

Different representations can serve different stages:

```text
Packed binary Hamming index
        ↓ inexpensive broad candidate retrieval
Top 100–1,000 candidates
        ↓ fp16/raw-count cosine reranking
Final top-k
        ↓ exact Cypher traversal
Graph-valid answers
```

This preserves magnitude-sensitive scoring only where it matters while keeping
the large first-stage index compact. The cost is storing or reconstructing the
reranking representation.

## Reduce the number of dimensions

Ten thousand dimensions is a common, convenient HDC scale, not a universal
requirement. Storage and almost all HDC compute scale linearly with dimension.

Benchmark smaller values such as 2,048, 4,096, or 8,192 against:

- Recall at `k`.
- Rank correlation with the 10,000-dimensional baseline.
- Threshold flips.
- Bundle size and cross-talk.
- Binding/unbinding recovery accuracy.
- Latency and bytes per entity.

If semantic text embeddings are projected upward into HDC space, remember that
the projection does not create new information. A larger target dimension makes
the binary angular estimate more stable and provides HDC capacity, but it should
still be selected empirically.

Search terms: **HDC dimensionality capacity**, **vector symbolic architecture
capacity**, **Johnson-Lindenstrauss random projection**, and **SimHash random
hyperplane LSH**.

## Consider sparse and structured HDC representations

Dense MAP is not the only HDC family. Alternatives include sparse binary
hypervectors, block codes, and structured or fast random projections. These may
reduce memory or compute, but they change the algebra and may not map directly
onto LanceDB's dense packed-binary index.

Search terms: **binary spatter codes**, **sparse distributed representations**,
**block-code hypervectors**, **sparse HDC**, **fast Johnson-Lindenstrauss
transform**, and **Hadamard random projection**.

## Control dataset and version retention

Columnar/versioned storage can retain superseded fragments and index versions.
After repeated re-encoding, dropping a column from the latest logical schema
does not always imply that every old physical byte has immediately disappeared.

For long-running datasets, investigate:

- Lance compaction.
- Old-version cleanup and retention policies.
- Index rebuild and replacement behavior.
- Fragment counts after repeated updates.
- Object-store lifecycle rules for backups and snapshots.

Search terms: **Lance compact files**, **Lance cleanup old versions**, and
**Lance dataset retention**.

## Avoid Python-object materialization during ingestion

This primarily reduces ingestion time and peak memory rather than final disk
size, but it matters at the same scales where storage becomes expensive.

The demo currently converts each tensor through `.tolist()`, creating 10,000
boxed Python floats per hypervector before Arrow casts the column to `float16`.
A production encoder should batch hypervectors and pass contiguous tensor,
NumPy, or Arrow buffers directly:

```text
batched Torch tensor
      ↓ native bulk conversion/cast
NumPy or Arrow float16 buffer
      ↓
Lance fixed-size-list column
```

This avoids millions or billions of short-lived Python objects and makes the
downcast a straightforward memory-bandwidth operation.

## A practical evaluation ladder

1. Materialize only hypervectors used by real queries.
2. Keep TorchHD computation in `float32`; store searchable bundles as `float16`.
3. Measure smaller HDC dimensions.
4. Add an SQ or RQ ANN index and evaluate recall with and without refinement.
5. Prototype packed binary majority bundles with Hamming search.
6. If binary recall is close, try binary first-stage retrieval plus fp16
   reranking.
7. Retain counters only for hypervector types that require incremental updates or
   magnitude-sensitive scoring.
8. Batch ingestion and enforce dataset/version retention policies.

For every experiment, record total bytes, index bytes, build time, query latency,
recall at `k`, rank correlation, and threshold flips. Compression is successful
only when it reduces the complete system cost without silently changing which
graph candidates reach the exact traversal stage.
