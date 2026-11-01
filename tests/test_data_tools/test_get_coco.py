"""
tests/test_data_tools/test_get_coco.py — COCO download: streaming, resume, restart
==================================================================================

A fake ``urlopen`` serves an in-memory payload; nothing is downloaded.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from data_tools import get_coco  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int, length: int | None) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {} if length is None else {"Content-Length": str(length)}


class TestDownload:
    payload = bytes(range(256)) * 40

    def _serve(self, monkeypatch, honour_range: bool) -> list[str | None]:
        seen: list[str | None] = []

        def fake_urlopen(req, timeout=None):
            rng = req.get_header("Range")
            seen.append(rng)
            if rng and honour_range:
                start = int(rng.split("=")[1].rstrip("-"))
                body = self.payload[start:]
                return FakeResponse(body, 206, len(body))
            return FakeResponse(self.payload, 200, len(self.payload))

        monkeypatch.setattr(get_coco.urllib.request, "urlopen", fake_urlopen)
        return seen

    def test_fresh_download(self, tmp_path, monkeypatch):
        self._serve(monkeypatch, honour_range=True)
        rec = get_coco.stream_download("http://x/f.zip", tmp_path / "f.zip", 1000)
        assert (tmp_path / "f.zip").read_bytes() == self.payload
        assert rec["sha256"] == hashlib.sha256(self.payload).hexdigest()
        assert rec["resumed_from"] == 0

    def test_resume_with_range(self, tmp_path, monkeypatch):
        seen = self._serve(monkeypatch, honour_range=True)
        (tmp_path / "f.zip.part").write_bytes(self.payload[:3000])
        rec = get_coco.stream_download("http://x/f.zip", tmp_path / "f.zip", 1000)
        assert seen == ["bytes=3000-"]
        assert rec["resumed_from"] == 3000
        assert (tmp_path / "f.zip").read_bytes() == self.payload
        assert rec["sha256"] == hashlib.sha256(self.payload).hexdigest()
        assert not (tmp_path / "f.zip.part").exists()

    def test_restart_without_range_support(self, tmp_path, monkeypatch):
        self._serve(monkeypatch, honour_range=False)
        (tmp_path / "f.zip.part").write_bytes(b"stale bytes")
        rec = get_coco.stream_download("http://x/f.zip", tmp_path / "f.zip", 1000)
        assert rec["resumed_from"] == 0
        assert (tmp_path / "f.zip").read_bytes() == self.payload

    def test_short_body_fails(self, tmp_path, monkeypatch):
        def fake_urlopen(req, timeout=None):
            return FakeResponse(self.payload[:100], 200, len(self.payload))

        monkeypatch.setattr(get_coco.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(get_coco.DownloadError):
            get_coco.stream_download("http://x/f.zip", tmp_path / "f.zip", 1000)
        assert not (tmp_path / "f.zip").exists()

    def test_blocked_download_stops(self, tmp_path, monkeypatch):
        def blocked(req, timeout=None):
            raise get_coco.urllib.error.URLError("blocked")

        monkeypatch.setattr(get_coco.urllib.request, "urlopen", blocked)
        raw = yaml.safe_load((REPO / "configs" / "coco.yaml").read_text(encoding="utf-8"))
        raw = copy.deepcopy(raw)
        raw["arena_config"] = str(REPO / "configs" / "arena.yaml")
        raw["paths"]["root"] = str(tmp_path / "coco")
        cfg_path = tmp_path / "coco.yaml"
        cfg_path.write_text(yaml.safe_dump(raw))
        assert get_coco.main(["--config", str(cfg_path)]) == 3
        assert not list((tmp_path / "coco").glob("*.zip"))

    def test_extract_refuses_unsafe_names(self, tmp_path):
        zpath = tmp_path / "a.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("../evil.json", "{}")
        with pytest.raises(ValueError):
            get_coco.extract_members(zpath, ["../evil.json"], tmp_path / "out")


MEMBERS = {
    "annotations/instances_val2017.json": b'{"images": []}',
    "annotations/captions_val2017.json": b"{}",
}


def ann_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in MEMBERS.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestVerifyAndRecord:
    def test_verify_member_rejects_crc_mismatch(self, tmp_path):
        zpath = tmp_path / "a.zip"
        zpath.write_bytes(ann_zip_bytes())
        with zipfile.ZipFile(zpath) as zf:
            info = zf.getinfo("annotations/instances_val2017.json")
        bad = tmp_path / "bad.json"
        bad.write_bytes(b'{"images": [}')  # same size, other bytes
        with pytest.raises(ValueError, match="CRC32"):
            get_coco.verify_member(bad, info)

    def test_tampered_existing_file_is_extracted_again(self, tmp_path):
        zpath = tmp_path / "a.zip"
        zpath.write_bytes(ann_zip_bytes())
        out = tmp_path / "out"
        target = out / "annotations" / "instances_val2017.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b'{"images": [}')
        res = get_coco.extract_members(zpath, list(MEMBERS), out)
        assert target.read_bytes() == MEMBERS["annotations/instances_val2017.json"]
        r = res["annotations/instances_val2017.json"]
        assert r["crc32_and_size_match_zip_directory"]
        assert r["sha256_extracted"] == hashlib.sha256(target.read_bytes()).hexdigest()
        assert r["uncompressed_bytes_zip_directory"] == len(
            MEMBERS["annotations/instances_val2017.json"]
        )

    def test_attempts_resumes_and_testzip_recorded(self, tmp_path, monkeypatch):
        payload = ann_zip_bytes()
        url = "http://x/annotations_trainval2017.zip"
        state = {"cut": True}

        def fake_urlopen(req, timeout=None):
            if req.get_method() == "HEAD":
                return FakeResponse(b"", 200, len(payload))
            rng = req.get_header("Range")
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            body = payload[start:]
            if state["cut"]:  # first attempt: the server sends half and closes
                state["cut"] = False
                return FakeResponse(body[: len(body) // 2], 206 if rng else 200, len(body))
            return FakeResponse(body, 206 if rng else 200, len(body))

        monkeypatch.setattr(get_coco.urllib.request, "urlopen", fake_urlopen)
        raw = yaml.safe_load((REPO / "configs" / "coco.yaml").read_text(encoding="utf-8"))
        raw["arena_config"] = str(REPO / "configs" / "arena.yaml")
        raw["paths"]["root"] = str(tmp_path / "coco")
        raw["download"]["urls"] = [url]
        raw["disk"]["min_free_gb"] = 0.0
        cfg_path = tmp_path / "coco.yaml"
        cfg_path.write_text(yaml.safe_dump(raw))

        assert get_coco.main(["--config", str(cfg_path)]) == 3
        rec = json.loads((tmp_path / "coco" / "DOWNLOAD.json").read_text())
        entry = rec["files"]["annotations_trainval2017.zip"]
        assert entry["attempts_count"] == 1 and entry["attempts"][0]["outcome"].startswith("failed")
        assert entry["attempts"][0]["end_byte"] == len(payload) // 2

        assert get_coco.main(["--config", str(cfg_path)]) == 0
        rec = json.loads((tmp_path / "coco" / "DOWNLOAD.json").read_text())
        entry = rec["files"]["annotations_trainval2017.zip"]
        assert (entry["attempts_count"], entry["resumes_count"]) == (2, 1)
        assert entry["resumed_across_attempts"] and entry["testzip_first_bad_member"] is None
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert set(rec["extracted"]) == set(MEMBERS)
        assert "not verified against an official checksum" in rec["note"]

        # A re-run keeps the attempt history and re-checks the extracted files
        assert get_coco.main(["--config", str(cfg_path)]) == 0
        rec = json.loads((tmp_path / "coco" / "DOWNLOAD.json").read_text())
        assert rec["files"]["annotations_trainval2017.zip"]["attempts_count"] == 2
