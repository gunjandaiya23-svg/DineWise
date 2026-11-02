#!/usr/bin/env python
"""P0 acceptance check: verify Python, GPU, packages, disk, and paths."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PACKAGES = [
    "torch",
    "transformers",
    "sentence_transformers",
    "qdrant_client",
    "pandas",
    "pyarrow",
    "yaml",
    "tqdm",
]


def main() -> int:
    report: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
        "cuda": {},
        "disk": {},
        "paths": {},
    }

    for name in PACKAGES:
        try:
            module = __import__(name)
            report["packages"][name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            report["packages"][name] = f"MISSING: {exc}"

    try:
        import torch

        available = torch.cuda.is_available()
        report["cuda"] = {
            "available": available,
            "device_count": torch.cuda.device_count(),
            "device": torch.cuda.get_device_name(0) if available else None,
            "torch_cuda": torch.version.cuda,
        }
        if available:
            props = torch.cuda.get_device_properties(0)
            report["cuda"]["total_memory_gb"] = round(props.total_memory / 1e9, 2)
            report["cuda"]["capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:  # noqa: BLE001
        report["cuda"] = {"available": False, "error": str(exc)}

    usage = shutil.disk_usage(ROOT)
    report["disk"] = {
        "total_gb": round(usage.total / 1e9, 1),
        "free_gb": round(usage.free / 1e9, 1),
    }
    report["paths"] = {
        name: str(ROOT / name) for name in ("config", "data", "docs", "src", "scripts")
    }
    report["paths_exist"] = {name: (ROOT / name).exists() for name in report["paths"]}

    try:
        from qdrant_client import QdrantClient

        from dinewise.config import load_config

        cfg = load_config()
        url = cfg["vector_store"]["url"]
        client = QdrantClient(url=url, timeout=3)
        collections = [item.name for item in client.get_collections().collections]
        report["qdrant"] = {
            "url": url,
            "reachable": True,
            "collections": collections,
        }
    except Exception as exc:  # noqa: BLE001
        report["qdrant"] = {"reachable": False, "error": str(exc)}

    output = ROOT / "results" / "env_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    missing = [name for name, value in report["packages"].items() if str(value).startswith("MISSING")]
    if missing:
        print(f"\nFAIL missing packages: {', '.join(missing)}")
        return 1
    if not report["cuda"].get("available"):
        print("\nFAIL CUDA is not available")
        return 1
    if not report["qdrant"].get("reachable"):
        print("\nFAIL Qdrant is not reachable; run `make db-up`")
        return 1
    print("\nOK environment is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
