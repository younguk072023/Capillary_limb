"""Focused inclusion and ICC checks for the GT-mask/predicted-U/D evaluation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import section4_gt_predud_agreement as report


def _sample_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = [f"p{i}_det_001.png" for i in range(1, 5)]
    selection = pd.DataFrame({"Filename": names, "category": ["u"] * 4,
                              "patient_id": [f"p{i}" for i in range(1, 5)]})
    manual = pd.DataFrame({
        "status": ["approved"] * 4,
        "manual_loop_px": [10.0, 20.0, 30.0, 40.0],
        "manual_thin_px": [10.0, 20.0, 30.0, 40.0],
        "manual_wide_px": [20.0, 30.0, 40.0, 50.0],
    }, index=names)
    export = pd.DataFrame({
        "fold": [1] * 4,
        "Status": ["Success", "Success", "Success", "Uncertain"],
        "Reason": ["", "", "", "low ray support"],
        "Loop_Diameter_px": [11.0, 21.0, 31.0, 41.0],
        "Branch1_Diameter_px": [11.0, 21.0, 31.0, 41.0],
        "Branch2_Diameter_px": [21.0, 31.0, 41.0, 51.0],
        "GT_Mask": ["gt.tif"] * 4,
        "Predicted_Mask": ["pred.png"] * 4,
        "Algorithm_SHA256": ["abc"] * 4,
    }, index=names)
    return selection, manual, export


def test_uncertain_branch_excluded_but_loop_kept() -> None:
    selection, manual, export = _sample_tables()
    pairs = report.make_pairs(selection, manual, export)
    assert len(pairs) == 4
    assert pairs["loop_pair"].sum() == 4
    assert pairs["branch_pair"].sum() == 3
    uncertain = pairs.iloc[3]
    assert uncertain["gt_reported_thin_px"] == 41.0
    assert math.isnan(uncertain["algorithm_thin_px"])
    assert uncertain["algorithm_loop_px"] == 41.0


def test_pearson_consistency_and_absolute_agreement_are_distinct() -> None:
    selection, manual, export = _sample_tables()
    pairs = report.make_pairs(selection, manual, export)
    agreement = report.make_agreement(pairs, bootstrap_reps=0)
    overall = agreement.loc[agreement["shape"] == "all"].set_index("target")
    assert int(overall.at["loop", "paired_n"]) == 4
    assert int(overall.at["thin", "paired_n"]) == 3
    assert int(overall.at["wide", "paired_n"]) == 3
    assert np.isclose(overall.at["thin", "pearson_r"], 1.0)
    assert np.isclose(overall.at["thin", "icc_consistency_C1"], 1.0)
    assert overall.at["thin", "icc_absolute_A1"] < 1.0
    assert np.isclose(overall.at["thin", "bias_px"], 1.0)


def test_changed_original_gt_hash_is_rejected(tmp_path) -> None:
    source = tmp_path / "mask.tif"
    source.write_bytes(b"changed")
    manifest = pd.DataFrame({
        "GT_Original": [str(source)], "Original_SHA256": ["0" * 64],
    }, index=["p1_det_001.png"])
    with pytest.raises(ValueError, match="hash changed"):
        report.verify_original_gt_sources(manifest)


@pytest.mark.parametrize("column,expected", [
    ("GT_Mask_SHA256", "GT or OOF mask hash mismatch"),
    ("Pred_U_x", "Predicted U/D changed"),
])
def test_real_export_provenance_tampering_is_rejected(
        tmp_path, column: str, expected: str) -> None:
    if not all(path.exists() for path in (
            report.DEFAULT_SELECTION, report.DEFAULT_MANUAL,
            report.DEFAULT_GT_MANIFEST, report.DEFAULT_OOF_RUN,
            report.DEFAULT_GT_EXPORT)):
        pytest.skip("Historical 144-image fixtures are unavailable")
    tampered = pd.read_csv(report.DEFAULT_GT_EXPORT, encoding="utf-8-sig")
    if column == "GT_Mask_SHA256":
        tampered.at[0, column] = "0" * 64
    else:
        tampered.at[0, column] = float(tampered.at[0, column]) + 1.0
    altered_csv = tmp_path / "altered_gt_export.csv"
    tampered.to_csv(altered_csv, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match=expected):
        report.evaluate(report.DEFAULT_SELECTION, report.DEFAULT_MANUAL,
                        report.DEFAULT_GT_MANIFEST, report.DEFAULT_OOF_RUN,
                        altered_csv, bootstrap_reps=0)
