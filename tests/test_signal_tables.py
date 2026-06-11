from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_processing.io import read_table_file  # noqa: E402
from fitting.signal_tables import normalize_signal_curve_keys, signal_table_analysis_id  # noqa: E402


class SignalTableTests(unittest.TestCase):
    def test_signal_table_analysis_id_strips_known_table_suffixes(self) -> None:
        self.assertEqual(signal_table_analysis_id("subj.rot_tensor.long.parquet"), "subj")
        self.assertEqual(signal_table_analysis_id("subj.long.parquet"), "subj")
        self.assertEqual(signal_table_analysis_id("subj.xlsx"), "subj")

    def test_normalize_signal_curve_keys_returns_groupable_rows(self) -> None:
        rows = pd.DataFrame(
            {
                "roi": [101],
                "direction": [2],
                "b_step": ["3"],
                "stat": ["avg"],
                "value_norm": [0.8],
            }
        )

        out = normalize_signal_curve_keys(rows, label="signal_rows")

        self.assertEqual(out.loc[0, "roi"], "101")
        self.assertEqual(out.loc[0, "direction"], "2")
        self.assertEqual(out.loc[0, "b_step"], 3)
        self.assertTrue(pd.api.types.is_integer_dtype(out["b_step"]))

    def test_read_table_file_reads_csv_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.csv"
            pd.DataFrame({"roi": ["ROI"], "value": [1.0]}).to_csv(path, index=False)

            out = read_table_file(path)

        self.assertEqual(out.to_dict(orient="records"), [{"roi": "ROI", "value": 1.0}])


if __name__ == "__main__":
    unittest.main()
