"""Run the final DINOv3 141-image batch export, including failed-sample overlays.

Uses the same workflow as predict.py so the two entry points cannot diverge.
Default: predicted masks. Options: --mask-source gt, --check, --output-dir PATH.
"""

import sys

sys.dont_write_bytecode = True
from predict import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
