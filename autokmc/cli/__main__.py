"""Support ``python -m autokmc.cli`` without pre-import warnings."""

from __future__ import annotations

import sys

from autokmc.cli.main import main


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
