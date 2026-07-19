from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import lance
import polars as pl
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hdc_encode import hv_merge_table, read_table
from torchhd_encoder import DIMENSIONS


class HdcEncodeTests(unittest.TestCase):
    def test_read_table_projects_out_blobs_and_existing_vectors(self) -> None:
        table = pa.table(
            {
                "id": ["row-1"],
                "name": ["Seattle"],
                "image": [b"image bytes"],
                "hv": pa.array(
                    [[1.0] * DIMENSIONS],
                    type=pa.list_(pa.float16(), DIMENSIONS),
                ),
                "vibe_hv": pa.array(
                    [[-1.0] * DIMENSIONS],
                    type=pa.list_(pa.float16(), DIMENSIONS),
                ),
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            db_uri = Path(temp_dir)
            lance.write_dataset(table, db_uri / "Location.lance")

            result = read_table(db_uri, "Location")

        self.assertEqual(result.columns, ["id", "name"])

    def test_hypervectors_are_stored_as_fixed_size_float16(self) -> None:
        rows = pl.DataFrame(
            {
                "id": ["row-1"],
                "hv": [[1.0] * DIMENSIONS],
            }
        )

        table = hv_merge_table(rows, ["hv"])

        self.assertEqual(
            table.schema.field("hv").type,
            pa.list_(pa.float16(), DIMENSIONS),
        )


if __name__ == "__main__":
    unittest.main()
