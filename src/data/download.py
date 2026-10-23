"""Download and verify the raw datasets.

The downloader is deliberately strict:

* it only fetches from the URLs declared in the dataset configuration,
* it stores the original archive unmodified,
* it records the size and the SHA-256 of every file,
* it records the license and citation information,
* it refuses to replace an existing file whose hash differs (a changed
  upstream archive is a *dataset version change* and must be handled
  explicitly, not silently).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..utils.checkpoints import sha256_file
from ..utils.logging import get_logger, setup_logging

__all__ = ["DatasetFile", "load_dataset_config", "download_file", "main"]

LOGGER = get_logger("data.download")


@dataclass
class DatasetFile:
    """One downloadable archive of a dataset."""

    name: str
    url: str
    size: int | None = None
    md5: str | None = None
    sha256: str | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetFile":
        return cls(
            name=str(data["name"]),
            url=str(data["url"]),
            size=int(data["size"]) if data.get("size") else None,
            md5=str(data["md5"]) if data.get("md5") else None,
            sha256=str(data["sha256"]) if data.get("sha256") else None,
            description=str(data.get("description", "")),
        )


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load a dataset configuration file."""
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"invalid dataset config: {path}")
    return config


def _md5_file(path: Path, chunk: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    file_spec: DatasetFile,
    target_dir: Path,
    *,
    overwrite: bool = False,
    timeout: int = 120,
) -> dict[str, Any]:
    """Download one archive and verify it.

    Returns:
        A metadata dictionary describing the stored file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file_spec.name

    if target.exists():
        digest = sha256_file(target)
        if file_spec.sha256 and digest != file_spec.sha256:
            if not overwrite:
                raise RuntimeError(
                    f"refusing to replace {target}: existing sha256={digest} "
                    f"does not match the expected {file_spec.sha256}. "
                    "Re-run with --overwrite only after confirming that the "
                    "upstream dataset version changed."
                )
            LOGGER.warning("overwriting %s because --overwrite was given", target)
            target.unlink()
        else:
            LOGGER.info("%s already present (sha256=%s)", target.name, digest[:16])
            return {
                "name": file_spec.name,
                "url": file_spec.url,
                "path": str(target),
                "size": target.stat().st_size,
                "sha256": digest,
                "md5": _md5_file(target),
                "downloaded_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                "status": "cached",
            }

    LOGGER.info("downloading %s -> %s", file_spec.url, target)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(file_spec.url, timeout=timeout) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        copied = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if total:
                pct = 100.0 * copied / total
                print(f"\r  {copied / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)", end="", flush=True)
    print()
    tmp.replace(target)

    digest = sha256_file(target)
    md5 = _md5_file(target)
    if file_spec.sha256 and digest != file_spec.sha256:
        raise RuntimeError(
            f"checksum mismatch for {file_spec.name}: got {digest}, expected {file_spec.sha256}"
        )
    if file_spec.md5 and md5 != file_spec.md5:
        raise RuntimeError(
            f"md5 mismatch for {file_spec.name}: got {md5}, expected {file_spec.md5}"
        )

    return {
        "name": file_spec.name,
        "url": file_spec.url,
        "path": str(target),
        "size": target.stat().st_size,
        "sha256": digest,
        "md5": md5,
        "downloaded_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "status": "downloaded",
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download and verify a dataset.")
    parser.add_argument("--config", required=True, help="dataset configuration YAML")
    parser.add_argument("--root", default="data/raw", help="root directory for raw data")
    parser.add_argument("--output", default=None, help="metadata output JSON")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing changed files")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))
    config = load_dataset_config(args.config)

    root = Path(args.root)
    download_dir = root / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    specs = [DatasetFile.from_dict(item) for item in config.get("files", [])]
    records: list[dict[str, Any]] = []
    for spec in specs:
        records.append(download_file(spec, download_dir, overwrite=args.overwrite))

    metadata = {
        "dataset": config.get("dataset", "unknown"),
        "source": config.get("source", ""),
        "source_url": config.get("source_url", ""),
        "record_id": config.get("record_id", ""),
        "license": config.get("license", ""),
        "citation": config.get("citation", ""),
        "downloaded_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "files": records,
    }

    out_path = Path(args.output) if args.output else download_dir / f"{config.get('dataset', 'dataset')}_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("wrote dataset metadata to %s", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
