"""Dataset download with checksum verification."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from tqdm import tqdm

from ..config import ROOT, data_path, ensure_dirs
from ..logging import get_logger

LOG = get_logger(__name__)
CHECKSUMS = ROOT / "config" / "dataset_checksums.json"
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected: str) -> None:
    temp = dest.with_suffix(dest.suffix + ".part")
    request = Request(url, headers={"User-Agent": "dinewise/0.1"})
    digest = hashlib.sha256()

    with urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with temp.open("wb") as handle, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as progress:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
                progress.update(len(block))

    actual = digest.hexdigest()
    if actual != expected:
        temp.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {dest.name}: {actual} != {expected}")

    temp.replace(dest)
    LOG.info("downloaded %s (%s)", dest.name, actual[:12])


def download_dataset(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Download every dataset file, skipping files that already verify."""
    ensure_dirs(cfg)
    spec = json.loads(CHECKSUMS.read_text(encoding="utf-8"))
    raw_dir = data_path(cfg, "raw_dir")
    manifest: dict[str, Any] = {
        "repo_id": spec["repo_id"],
        "downloaded_at": datetime.now(UTC).isoformat(),
        "files": {},
    }

    for name, expected in spec["files"].items():
        dest = raw_dir / name
        if dest.exists() and not force:
            actual = sha256_file(dest)
            if actual == expected:
                LOG.info("verified %s", name)
                manifest["files"][name] = {
                    "sha256": actual,
                    "bytes": dest.stat().st_size,
                    "status": "cached",
                }
                continue
            LOG.warning("%s exists but failed verification; re-downloading", name)

        url = f"{spec['base_url']}/{name}"
        _download(url, dest, expected)
        manifest["files"][name] = {
            "sha256": expected,
            "bytes": dest.stat().st_size,
            "status": "downloaded",
        }

    (raw_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
