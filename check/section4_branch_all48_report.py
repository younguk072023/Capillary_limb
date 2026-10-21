"""Write a concise 48-per-shape branch agreement TXT from stored raw measurements.

This does not rerun the geometry analyzer. It includes numeric measurements
flagged Uncertain while retaining their counts in the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CHECK_DIR = Path.home() / "Desktop" / "Capillary_limb" / "check"
if str(CHECK_DIR) not in sys.path:
    sys.path.insert(0, str(CHECK_DIR))

import section4_validation as validation


DEFAULT_PAIRS = (
    CHECK_DIR / "section4_gt_predud_agreement_pa_144_v1_precheck"
    / "section4_gt_predud_pairs.csv"
)
SHAPES = ("u", "tortuous", "twist")
TARGETS = (
    ("thin", "가는 분지(세동맥측 추정)"),
    ("wide", "굵은 분지(세정맥측 추정)"),
)
REQUIRED = (
    "Filename", "shape", "patient_id", "manual_status", "algorithm_status",
    "manual_thin_px", "manual_wide_px",
    "gt_reported_thin_px", "gt_reported_wide_px",
)


def make_report(pairs_path: Path) -> str:
    pairs = pd.read_csv(pairs_path, encoding="utf-8-sig")
    validation.require_columns(pairs, REQUIRED, pairs_path)
    validation.require_unique(pairs, "Filename", pairs_path)
    if len(pairs) != 144 or set(pairs["shape"]) != set(SHAPES):
        raise ValueError("Expected exactly 144 images in U, tortuous, and twist")
    if not pairs["manual_status"].eq("approved").all():
        raise ValueError("All manual measurements must be approved")
    if not pairs["algorithm_status"].isin(("Success", "Uncertain")).all():
        raise ValueError("All 144 images must have Success or Uncertain numeric results")
    for field in ("manual_thin_px", "manual_wide_px",
                  "gt_reported_thin_px", "gt_reported_wide_px"):
        values = pd.to_numeric(pairs[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Missing or nonpositive numeric width: {field}")
        pairs[field] = values
    pairs["status"] = pairs["manual_status"]

    provenance_path = pairs_path.with_name("section4_gt_predud_provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    run_id = provenance.get("run_id", "unknown")

    lines = [
        "제4절 분지 직경 비교 — 각 형태 48장 전체 포함 (사전 점검)",
        "",
        "수동값과 자동값: U형·tortuous·twist 각 48장, 총 144장",
        "자동값 입력: 복사된 GT 분할 마스크 + 기존 OOF MTL 예측 U/D",
        f"OOF 실행 ID: {run_id}",
        "루프 직경은 제외하며 최종 MTL 평가에서 별도로 비교함.",
        "",
        "형태\t측정 부위\t평가 n\tPearson r\tICC(C,1) 일관성\tICC(A,1) 절대일치",
    ]
    for shape in SHAPES:
        group = pairs.loc[pairs["shape"] == shape]
        if len(group) != 48:
            raise ValueError(f"Expected 48 selected images for {shape}, got {len(group)}")
        for target, label in TARGETS:
            result = validation.summary_row(
                group, target, shape,
                f"manual_{target}_px", f"gt_reported_{target}_px",
                bootstrap_reps=0,
            )
            if result["paired_n"] != 48:
                raise ValueError(f"Expected 48 valid pairs for {shape}/{target}")
            lines.append("\t".join((
                "U형" if shape == "u" else shape,
                label,
                "48",
                f'{result["pearson_r"]:.3f}',
                f'{result["icc_consistency_C1"]:.3f}',
                f'{result["icc_absolute_A1"]:.3f}',
            )))
    warning_counts = [
        f'{"U형" if shape == "u" else shape} {int(pairs.loc[pairs["shape"] == shape, "algorithm_status"].eq("Uncertain").sum())}장'
        for shape in SHAPES
    ]
    lines += [
        "",
        "주: 숫자가 산출된 Uncertain 사례도 통계에 포함함. " + ", ".join(warning_counts) + ".",
        "    Uncertain은 자동 측정선 품질 경고이며 수동 검증으로 오류가 확정된 것은 아님.",
        "    가는/굵은 분지는 직경 크기에 따른 세동맥측/세정맥측 추정 명칭임.",
        "    현재 값은 GT 분할 + 기존 OOF 예측 U/D의 사전 점검이며 최종 MTL 결과는 아님.",
        f"자료: {pairs_path.name}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.pairs.with_name("section4_branch_metrics_simple_precheck.txt")
    report = make_report(args.pairs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8-sig")
    print(f"Saved all-48 branch metrics: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
