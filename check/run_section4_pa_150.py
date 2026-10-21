"""Run the Section 4 OOF diameter exporter on the selected 150 images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = (Path.home() / "Desktop" / "Multi-Task-Learning" /
               "results_unet_mtl_heatmap_bce_dice_5849db92da83")
DEFAULT_SELECTION = Path.home() / "Desktop" / "Pa_capillary"
DEFAULT_OUTPUT = HERE / "section4_oof_limb_pa_150_v3.csv"
CV2_SITE = Path.home() / ".conda" / "envs" / "younguk" / "Lib" / "site-packages"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # The existing base environment has scipy/scikit-image. The installed
    # OpenCV package is in younguk; append its site-packages after base paths
    # so base numpy/scipy are not shadowed.
    if not args.dry_run:
        if not CV2_SITE.is_dir():
            parser.error(f"Existing OpenCV package directory is missing: {CV2_SITE}")
        sys.path.append(str(CV2_SITE))
        try:
            import cv2  # noqa: F401
            import skimage  # noqa: F401
        except ImportError as exc:
            parser.error(f"Analyzer dependency is missing: {exc}")

    from section4_oof_limb_export import main as export_main

    command = ["--run-dir", str(args.run_dir),
               "--selection-dir", str(args.selection_dir),
               "--expected-count", "150",
               "--output", str(args.output)]
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    return export_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
