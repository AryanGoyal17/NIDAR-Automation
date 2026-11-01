"""
data_tools/get_coco.py — Setup-time download of COCO val2017 (images + annotations)
===================================================================================

COCO sanity set, not arena. The only script in this repo allowed to use the
internet, and only at setup time. Downloads the two zips listed in
``configs/coco.yaml`` into ``data/coco/``:

- streams each file in chunks (never the whole zip in memory) to ``<name>.part``
  and hashes it while streaming;
- resumes a ``.part`` file with an HTTP Range request when the server answers
  206; on a plain 200 it restarts from zero;
- renames ``.part`` to the final name when the byte count matches
  Content-Length, and records url, bytes, sha256 and UTC time in
  ``data/coco/DOWNLOAD.json``. COCO publishes no checksum we know of, so the
  sha256 is recorded, not verified (TBD);
- records every attempt (start byte, end byte, outcome) in ``DOWNLOAD.json``,
  failed ones included, with ``attempts_count`` and ``resumes_count``;
- skips a file that already exists (re-hashes it; a sha256 different from the
  recorded one is an error);
- runs ``zipfile.testzip`` on each zip (CRC32 of every member) and records the result;
- extracts only the annotation members listed in the config and checks each
  one against the CRC32 and size in the zip directory (a mismatch is an error);
  records CRC32, sizes and the sha256 of the extracted file. ``val2017.zip``
  stays zipped; the other tools read images from it by name.

If a download is blocked or fails, the script stops with exit code 3. It never
falls back to another source or dataset.

Usage::

    python -m data_tools.get_coco
    python -m data_tools.get_coco --config configs/coco.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from cv.logger import get_logger
from data_tools.coco_common import (
    GB,
    CocoConfig,
    DiskSpaceError,
    atomic_write_json,
    check_disk,
    free_bytes,
    sha256_file,
)

log = get_logger("get_coco", log_file=None)

DOWNLOAD_RECORD = "DOWNLOAD.json"
TIMEOUT_S = 60


class DownloadError(RuntimeError):
    """A download was blocked, failed or came back with the wrong size."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def remote_size(url: str) -> int | None:
    """Content-Length from a HEAD request, or None if the server does not say."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        length = resp.headers.get("Content-Length")
    return int(length) if length is not None else None


def stream_download(url: str, dest: Path, chunk_bytes: int) -> dict[str, Any]:
    """Download ``url`` to ``dest`` via ``dest.part``; resume when the server allows it.

    Returns the record (bytes, sha256, resumed_from). Raises DownloadError.
    """
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    h = hashlib.sha256()
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:  # range not satisfiable: restart cleanly
            part.unlink()
            return stream_download(url, dest, chunk_bytes)
        raise DownloadError(f"{url}: HTTP {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DownloadError(f"{url}: {exc}") from exc

    with resp:
        status = getattr(resp, "status", 200)
        length = resp.headers.get("Content-Length")
        if have and status == 206:
            # Hash what is already on disk, then append
            with open(part, "rb") as f:
                while block := f.read(chunk_bytes):
                    h.update(block)
            mode, resumed_from = "ab", have
            expected = have + int(length) if length is not None else None
            log.info(f"resuming {dest.name} at {have} bytes")
        else:
            if have:
                log.info(
                    f"server ignored the range request (HTTP {status}); restarting {dest.name}"
                )
            mode, resumed_from = "wb", 0
            expected = int(length) if length is not None else None
        written = resumed_from
        try:
            with open(part, mode) as f:
                while block := resp.read(chunk_bytes):
                    f.write(block)
                    h.update(block)
                    written += len(block)
        except OSError as exc:
            raise DownloadError(f"{url}: interrupted after {written} bytes ({exc})") from exc

    if expected is not None and written != expected:
        raise DownloadError(f"{url}: got {written} bytes, expected {expected}; re-run to resume")
    part.replace(dest)
    return {"bytes": written, "sha256": h.hexdigest(), "resumed_from": resumed_from}


def verify_member(path: Path, info: zipfile.ZipInfo) -> dict[str, Any]:
    """Check an extracted file against the zip directory (size, CRC32); raise on mismatch."""
    crc, size, h = 0, 0, hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            crc = zlib.crc32(block, crc)
            size += len(block)
            h.update(block)
    if size != info.file_size or crc != info.CRC:
        raise ValueError(
            f"{path}: size {size} / CRC32 {crc:08x} do not match the zip directory "
            f"({info.file_size} / {info.CRC:08x})"
        )
    return {
        "crc32_zip_directory": f"{info.CRC:08x}",
        "uncompressed_bytes_zip_directory": info.file_size,
        "compressed_bytes_zip_directory": info.compress_size,
        "sha256_extracted": h.hexdigest(),
        "crc32_and_size_match_zip_directory": True,
    }


def extract_members(
    zip_path: Path, members: list[str], out_root: Path
) -> dict[str, dict[str, Any]]:
    """Stream the listed members out of a zip into ``out_root`` and verify each one.

    Refuses unsafe names. A file that already exists is kept only if its size and
    CRC32 match the zip directory; otherwise it is extracted again. A mismatch
    after extraction raises ValueError.
    """
    results = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in members:
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe zip member name: {name}")
            info = zf.getinfo(name)
            dest = out_root.joinpath(*rel.parts)
            if dest.exists():
                try:
                    results[name] = verify_member(dest, info)
                    log.info(f"exists, CRC32 and size match the zip directory: {dest}")
                    continue
                except ValueError as exc:
                    log.warning(f"{exc}; extracting again")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            with zf.open(name) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            results[name] = verify_member(tmp, info)
            tmp.replace(dest)
            log.info(f"extracted {name} -> {dest}")
    return results


def first_bad_member(path: Path) -> str | None:
    """First member whose CRC fails (zipfile.testzip), or None if all members are good."""
    with zipfile.ZipFile(path) as zf:
        return zf.testzip()


def _part_size(dest: Path) -> int:
    part = dest.with_name(dest.name + ".part")
    return part.stat().st_size if part.exists() else 0


def _summarise_attempts(entry: dict[str, Any]) -> None:
    attempts = entry.get("attempts", [])
    entry["attempts_count"] = len(attempts)
    entry["resumes_count"] = sum(1 for a in attempts if a.get("start_byte", 0) > 0)
    entry["resumed_across_attempts"] = entry["resumes_count"] > 0


def get_coco(cfg: CocoConfig) -> dict[str, Any]:
    root = Path(cfg["paths"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / DOWNLOAD_RECORD
    record: dict[str, Any] = {"files": {}}
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
    chunk = int(cfg["download"]["chunk_bytes"])
    log.info(f"free space before downloads: {free_bytes(root) / GB:.2f} GB")

    for url in cfg["download"]["urls"]:
        name = url.rsplit("/", 1)[-1]
        dest = root / name
        entry = dict(record["files"].get(name, {}))
        entry["url"] = url
        if dest.exists():
            log.info(f"exists, skipped download: {dest}; hashing for the record")
            sha = sha256_file(dest)
            if entry.get("sha256") and entry["sha256"] != sha:
                raise DownloadError(
                    f"{dest}: sha256 {sha} differs from the recorded {entry['sha256']}"
                )
            entry.update(bytes=dest.stat().st_size, sha256=sha, rehashed_utc=_utc_now())
        else:
            try:
                size = remote_size(url)
            except (urllib.error.URLError, OSError) as exc:
                raise DownloadError(f"{url}: {exc}") from exc
            remaining = (size or 0) - _part_size(dest)
            free = check_disk(root, max(0, remaining), cfg, f"download {name}")
            log.info(
                f"downloading {url} ({size} bytes, {free / GB:.2f} GB free before; "
                f"{(free - max(0, remaining)) / GB:.2f} GB expected after)"
            )
            attempt = {"utc": _utc_now(), "start_byte": _part_size(dest)}
            entry.setdefault("attempts", []).append(attempt)
            try:
                result = stream_download(url, dest, chunk)
            except DownloadError as exc:
                attempt.update(end_byte=_part_size(dest), outcome=f"failed: {exc}")
                _summarise_attempts(entry)
                record["files"][name] = entry
                atomic_write_json(record_path, record)
                raise
            attempt.update(end_byte=result["bytes"], outcome="complete")
            entry.update(bytes=result["bytes"], sha256=result["sha256"], downloaded_utc=_utc_now())
            log.info(f"downloaded {name}: {entry['bytes']} bytes, sha256 {entry['sha256']}")
        _summarise_attempts(entry)
        bad = first_bad_member(dest)
        entry.update(testzip_first_bad_member=bad, testzip_utc=_utc_now())
        entry.pop("resumed_from", None)
        entry.pop("skipped_existing_utc", None)
        record["files"][name] = entry
        atomic_write_json(record_path, record)
        if bad is not None:
            raise DownloadError(f"{dest}: zip member {bad} fails its CRC check")

    ann_zip = root / "annotations_trainval2017.zip"
    members = list(cfg["download"]["extract"])
    with zipfile.ZipFile(ann_zip) as zf:
        need = sum(
            zf.getinfo(m).file_size for m in members if not root.joinpath(*m.split("/")).exists()
        )
    check_disk(root, need, cfg, "extract annotations")
    record["extracted"] = {
        m: dict(r, source_zip=ann_zip.name)
        for m, r in extract_members(ann_zip, members, root).items()
    }
    record["note"] = (
        "sha256 values are recorded, not verified against an official checksum (COCO publishes "
        "none we know of; TBD). Downloads may be resumed across several attempts (HTTP Range); "
        "attempts_count and resumes_count say how many. testzip checks the CRC32 of every member "
        "inside a zip; extracted files are checked against the CRC32 and size in the zip "
        "directory. COCO sanity set, not arena."
    )
    atomic_write_json(record_path, record)
    log.info(f"free space after: {free_bytes(root) / GB:.2f} GB; record: {record_path}")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default="configs/coco.yaml")
    args = ap.parse_args(argv)
    cfg = CocoConfig.from_yaml(args.config)
    try:
        get_coco(cfg)
    except DownloadError as exc:
        log.error(f"download failed or blocked: {exc}")
        log.error("stopping; no other source or dataset is used as a substitute")
        return 3
    except DiskSpaceError as exc:
        log.error(str(exc))
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
