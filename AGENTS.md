# Repository notes

- This is a Python 3.13 `uv` project. Run tests with
  `uv run python -m unittest discover -s tests -v` and the end-to-end demo
  with `uv run python src/run_person_location_demo.py`.
- `graph_ingest.py` creates the `Person`, `Location`, and `VISITED` Lance tables;
  `hdc_encode.py` adds HDC columns to those same tables. Fuzzy retrieval searches
  `Location.vibe_hv`, then lance-graph/Cypher expands candidates through exact
  `VISITED` edges.
- HDC uses deterministic 10,000-dimensional TorchHD MAP vectors. Preserve raw
  bundle sums; normalize composite vectors to bipolar `{-1, +1}` only before
  binding.
- TorchHD algebra must remain `float32`. Persisted Lance vector columns are
  intentionally `float16` to halve storage. TorchHD does not officially support
  MAP `float16`: always upcast stored vectors to `float32` with `list_to_hv()` (or
  an equivalent explicit cast) before TorchHD operations. Downcast only at the
  Arrow/Lance storage boundary in `hv_merge_table()`.
- Keep image blob columns lazy and out of encoding/query projections. Avoid
  materializing hypervectors as Python lists in any new scale-oriented path;
  prefer batched tensor/NumPy/Arrow buffers.
