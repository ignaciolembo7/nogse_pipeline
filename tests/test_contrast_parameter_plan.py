from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ogse_fitting.contrast_parameter_plan import (  # noqa: E402
    contrast_parameter_vary_flags,
    fixed_contrast_parameter_values,
    normalize_global_contrast_params,
    seed_bounds_for_contrast_parameter,
)


class ContrastParameterPlanTests(unittest.TestCase):
    def test_normalize_global_contrast_params_resolves_physical_aliases(self) -> None:
        self.assertEqual(normalize_global_contrast_params("mixed", ["tc,D0", "M0"]), ["tc_ms", "D0_m2_ms", "M0"])
        self.assertEqual(normalize_global_contrast_params("mixed", ["NONE"]), [])
        self.assertEqual(normalize_global_contrast_params("mixed", ["LOCAL"]), [])

    def test_normalize_global_contrast_params_rejects_param_not_in_model(self) -> None:
        with self.assertRaises(ValueError):
            normalize_global_contrast_params("free", ["tc"])

    def test_seed_bounds_for_contrast_parameter_clips_initial_value(self) -> None:
        seed, lo, hi = seed_bounds_for_contrast_parameter(
            "tc_ms",
            M0_value=1.0,
            D0_value=2.3e-12,
            C_value=0.0,
            tc_value=2000.0,
            tc_bounds=(0.1, 1000.0),
            m0_bounds=None,
            d0_bounds=None,
            c_bounds=None,
        )

        self.assertEqual((seed, lo, hi), (1000.0, 0.1, 1000.0))

    def test_fixed_and_vary_values_are_named_by_physical_parameter(self) -> None:
        self.assertEqual(contrast_parameter_vary_flags(M0_vary=True, D0_vary=False, C_vary=True, tc_vary=False)["C"], True)
        fixed = fixed_contrast_parameter_values(M0_value=1.0, D0_value=2.3e-12, C_value=0.02, tc_value=5.0)
        self.assertEqual(fixed["tc_ms"], 5.0)
        self.assertEqual(fixed["D0_m2_ms"], 2.3e-12)


if __name__ == "__main__":
    unittest.main()
