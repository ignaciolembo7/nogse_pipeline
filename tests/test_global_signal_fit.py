from __future__ import annotations

import unittest

import numpy as np

from fitting.global_signal_fit import scope_members, scope_name
from fitting.global_signal_inputs import CurveData
from fitting.parameter_modes import FitParameterConfig


def _curve(curve_id: int, pair_key: str) -> CurveData:
    return CurveData(
        curve_id=curve_id,
        source_file=f"source_{curve_id}.parquet",
        analysis_id=f"analysis_{curve_id}",
        subj="S1",
        roi="ROI",
        direction="long",
        stat="avg",
        td_ms=90.0,
        n_value=4.0,
        x_model_ms=22.5,
        sequence="",
        sheet="",
        protocol="",
        contrast_analysis_id="",
        contrast_source_file="",
        contrast_side=0,
        contrast_N_1=np.nan,
        contrast_N_2=np.nan,
        pair_key=pair_key,
        g=np.array([0.0, 10.0]),
        f_corr=1.0,
        corr_status="not_requested",
        y=np.array([1.0, 0.8]),
        b_step=np.array([0.0, 1.0]),
    )


def _config(name: str, mode: str) -> FitParameterConfig:
    return FitParameterConfig(name=name, mode=mode, init=1.0, fixed=None, bounds=(0.0, 10.0))


class GlobalSignalFitModeTests(unittest.TestCase):
    def test_parameter_modes_define_physical_sharing_scopes(self) -> None:
        curves = [_curve(1, "pair_a"), _curve(2, "pair_a"), _curve(3, "pair_b")]
        pair_param_ids = {"pair_a": 1, "pair_b": 3}

        self.assertIsNone(scope_name(_config("M0", "fixed"), curves[0], pair_param_ids))
        self.assertEqual(scope_name(_config("tc_ms", "global_td"), curves[0], pair_param_ids), "tc_ms")
        self.assertEqual(scope_name(_config("M0", "global_contrast"), curves[1], pair_param_ids), "M0__pair_1")
        self.assertEqual(scope_name(_config("M0", "global_contrast"), curves[2], pair_param_ids), "M0__pair_3")
        self.assertEqual(scope_name(_config("C", "free"), curves[2], pair_param_ids), "C__curve_3")

        global_members = scope_members(curves, config=_config("tc_ms", "global_td"), pair_param_ids=pair_param_ids)
        contrast_members = scope_members(curves, config=_config("M0", "global_contrast"), pair_param_ids=pair_param_ids)
        free_members = scope_members(curves, config=_config("C", "free"), pair_param_ids=pair_param_ids)

        self.assertEqual([(name, [c.curve_id for c in members]) for name, members in global_members], [("tc_ms", [1, 2, 3])])
        self.assertEqual(
            [(name, [c.curve_id for c in members]) for name, members in contrast_members],
            [("M0__pair_1", [1, 2]), ("M0__pair_3", [3])],
        )
        self.assertEqual(
            [(name, [c.curve_id for c in members]) for name, members in free_members],
            [("C__curve_1", [1]), ("C__curve_2", [2]), ("C__curve_3", [3])],
        )


if __name__ == "__main__":
    unittest.main()
