"""Create a visual review queue without changing any diameter measurements."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from section4_qc_visuals import render, rows_by_filename


def number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return float(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-csv", type=Path, required=True)
    parser.add_argument("--algorithm-csv", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    if args.top_n < 0:
        parser.error("--top-n must be nonnegative")

    manual = rows_by_filename(args.manual_csv)
    predicted = rows_by_filename(args.algorithm_csv)
    if set(manual) != set(predicted):
        parser.error("Manual and algorithm filename sets do not match")

    errors: list[tuple[float, str]] = []
    for filename, row in predicted.items():
        if row["Status"] != "Success" or manual[filename].get("status") != "approved":
            continue
        thin = number(row, "Estimated_Arterial_Diameter_px")
        wide = number(row, "Estimated_Venous_Diameter_px")
        manual_thin = number(manual[filename], "manual_thin_px")
        manual_wide = number(manual[filename], "manual_wide_px")
        if None not in (thin, wide, manual_thin, manual_wide):
            errors.append((max(abs(thin - manual_thin), abs(wide - manual_wide)), filename))
    selected = sorted(filename for filename, row in predicted.items()
                      if row["Status"] != "Success")
    selected.extend(filename for _, filename in sorted(errors, reverse=True)[:args.top_n])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = args.output_dir / "review_queue.csv"
    fields = ["Filename", "shape", "Status", "Reason", "Review_Trigger",
              "Manual_Thin_px", "Auto_Thin_px", "Abs_Thin_Error_px",
              "Manual_Wide_px", "Auto_Wide_px", "Abs_Wide_Error_px",
              "Branch_Path_Balance", "Overlay"]
    with summary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for filename in selected:
            m = manual[filename]
            p = predicted[filename]
            overlay = render(m, p, args.selection_dir, args.output_dir)
            thin, wide = number(p, "Estimated_Arterial_Diameter_px"), number(p, "Estimated_Venous_Diameter_px")
            manual_thin, manual_wide = number(m, "manual_thin_px"), number(m, "manual_wide_px")
            writer.writerow({
                "Filename": filename, "shape": m["shape"],
                "Status": p["Status"], "Reason": p["Reason"],
                "Review_Trigger": "Algorithm uncertainty" if p["Status"] != "Success" else "Large manual difference",
                "Manual_Thin_px": manual_thin, "Auto_Thin_px": thin,
                "Abs_Thin_Error_px": abs(thin - manual_thin) if None not in (thin, manual_thin) else "",
                "Manual_Wide_px": manual_wide, "Auto_Wide_px": wide,
                "Abs_Wide_Error_px": abs(wide - manual_wide) if None not in (wide, manual_wide) else "",
                "Branch_Path_Balance": p.get("Branch_Path_Balance", ""),
                "Overlay": str(overlay),
            })
    print(f"Saved {len(selected)} review overlays and {summary}")


if __name__ == "__main__":
    main()
