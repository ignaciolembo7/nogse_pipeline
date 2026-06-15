from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts" / "data"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from filter_master_table import filter_first_points_by_td, parse_first_points_by_td  # noqa: E402


class FilterMasterTableTests(unittest.TestCase):
    def test_parse_first_points_by_td_accepts_all(self) -> None:
        parsed = parse_first_points_by_td("120=8,210=6 90=ALL")

        self.assertEqual(parsed, {120.0: 8, 210.0: 6, 90.0: None})

    def test_filter_first_points_by_td_keeps_unlisted_td(self) -> None:
        rows = pd.DataFrame(
            {
                "row_kind": ["signal"] * 8,
                "stat": ["avg"] * 8,
                "roi": ["ROI_A"] * 8,
                "direction": ["long"] * 8,
                "b_step": [0, 1, 2, 3, 0, 1, 2, 3],
                "bvalue": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
                "value": [1.0] * 8,
                "td_ms": [120.0] * 4 + [90.0] * 4,
            }
        )

        selected = filter_first_points_by_td(rows, "120=2")

        self.assertEqual(selected[selected["td_ms"] == 120.0]["b_step"].tolist(), [0, 1])
        self.assertEqual(selected[selected["td_ms"] == 90.0]["b_step"].tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
