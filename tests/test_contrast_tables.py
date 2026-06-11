from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fitting.contrast_tables import (  # noqa: E402
    analysis_id_from_source_file,
    coerce_correction_pair,
    fit_row_correction_pair,
    normalize_contrast_keys,
    require_columns,
    unique_scalar,
)


class ContrastTableTests(unittest.TestCase):
    def test_normalize_contrast_keys_keeps_curve_keys_stable(self) -> None:
        rows = pd.DataFrame(
            {
                "roi": ["ROI"],
                "direction": [1],
                "b_step": ["2"],
                "stat": ["avg"],
                "value": [0.2],
            }
        )

        out = normalize_contrast_keys(rows, label="contrast_rows")

        self.assertEqual(out.loc[0, "direction"], "1")
        self.assertEqual(out.loc[0, "b_step"], 2)
        self.assertTrue(pd.api.types.is_integer_dtype(out["b_step"]))

    def test_normalize_contrast_keys_rejects_bad_b_step(self) -> None:
        rows = pd.DataFrame({"roi": ["ROI"], "direction": ["long"], "b_step": ["bad"]})

        with self.assertRaises(ValueError):
            normalize_contrast_keys(rows, label="contrast_rows")

    def test_require_columns_names_missing_physical_columns(self) -> None:
        with self.assertRaises(KeyError):
            require_columns(pd.DataFrame({"roi": ["ROI"]}), ["roi", "direction"], label="curve")

    def test_unique_scalar_rejects_mixed_group_values(self) -> None:
        self.assertEqual(unique_scalar(pd.Series([3, 3, np.nan]), name="N", required=True), 3)
        with self.assertRaises(ValueError):
            unique_scalar(pd.Series([3, 4]), name="N", required=True)

    def test_analysis_id_comes_from_signal_table_name(self) -> None:
        self.assertEqual(analysis_id_from_source_file("subject.long.parquet"), "subject")
        self.assertEqual(analysis_id_from_source_file("subject.parquet"), "subject")
        self.assertEqual(analysis_id_from_source_file(None), "")

    def test_correction_pair_returns_positive_two_side_factors(self) -> None:
        self.assertEqual(coerce_correction_pair(None), (1.0, 1.0))
        self.assertEqual(coerce_correction_pair(1.2), (1.2, 1.2))
        self.assertEqual(coerce_correction_pair((1.2, 0.8)), (1.2, 0.8))
        self.assertEqual(coerce_correction_pair((0.0, np.nan)), (1.0, 1.0))
        self.assertEqual(fit_row_correction_pair({"f_corr_1": 1.1, "f_corr_2": 0.9}), (1.1, 0.9))


if __name__ == "__main__":
    unittest.main()
