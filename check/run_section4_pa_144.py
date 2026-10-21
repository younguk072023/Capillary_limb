"""Export the selected DINOv3 OOF diameter measurements (141 images by default).

The historical filename is retained for existing shortcuts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

DESKTOP = Path.home() / "Desktop"
TARGET_CHECK = DESKTOP / "Capillary_limb" / "check"
DEFAULT_RUN = DESKTOP / "논문 결과" / "results_unet_mtl_dinov3_vitb16_finetune_heatmap_dbc65ae36d7e"
DEFAULT_SELECTION_CSV = TARGET_CHECK / "diameter_selection_141.csv"
DEFAULT_OUTPUT = TARGET_CHECK / "section4_oof_limb_dinov3_141.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--selection-dir", type=Path)
    selection.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--expected-count", type=int, default=141)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    if str(TARGET_CHECK) not in sys.path:
        sys.path.insert(0, str(TARGET_CHECK))
    from section4_oof_limb_export import main as export_main
    command = ["--run-dir", str(args.run_dir), "--expected-count", str(args.expected_count),
               "--output", str(args.output)]
    if args.selection_dir is not None:
        command.extend(["--selection-dir", str(args.selection_dir)])
    else:
        command.extend(["--selection-csv", str(args.selection_csv)])
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    return export_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
