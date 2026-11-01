#!/usr/bin/env python
"""Download the dataset and verify checksums."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dinewise.config import load_config  # noqa: E402
from dinewise.data.download import download_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="redownload existing files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = download_dataset(cfg, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
