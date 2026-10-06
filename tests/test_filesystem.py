"""Filesystem tool tests using a temp dir inside ALLOWED_ROOTS."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _sandbox(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="lff_")
    # Point allowed roots + index DB at the sandbox.
    monkeypatch.setenv("ALLOWED_ROOTS", tmp)
    monkeypatch.setenv("FILE_INDEX_DB", os.path.join(tmp, "_idx.db"))
    import importlib
    import app.config as cfg

    importlib.reload(cfg)
    import app.tools.filesystem as fs

    importlib.reload(fs)
    return tmp, fs


def test_search_finds_real_file(monkeypatch):
    tmp, fs = _sandbox(monkeypatch)
    Path(tmp, "identity_document.pdf").write_text("dummy")
    Path(tmp, "notes.txt").write_text("dummy")
    res = fs.search_files_core(tmp, "identity document", limit=5)
    paths = [r["path"] for r in res]
    assert any("identity_document.pdf" in p for p in paths)


def test_search_extension_hint(monkeypatch):
    tmp, fs = _sandbox(monkeypatch)
    Path(tmp, "report.pdf").write_text("x")
    Path(tmp, "report.txt").write_text("x")
    res = fs.search_files_core(tmp, "report pdfs dhoondo", limit=5)
    assert res, "expected at least one candidate"
    assert res[0]["path"].lower().endswith(".pdf")


def test_nonexistent_query_returns_empty(monkeypatch):
    tmp, fs = _sandbox(monkeypatch)
    Path(tmp, "hello.txt").write_text("x")
    res = fs.search_files_core(tmp, "aesi file jo exist hi nahi karti zzzqxq", limit=5)
    assert res == []


def test_list_and_info(monkeypatch):
    tmp, fs = _sandbox(monkeypatch)
    f = Path(tmp, "a.txt")
    f.write_text("hi")
    listing = fs.list_directory_core(tmp)
    assert any(e["name"] == "a.txt" for e in listing)
    info = fs.get_file_info_core(str(f))
    assert info["filename"] == "a.txt"


def test_no_hardcoded_synonym_map():
    src = Path(ROOT, "app", "tools", "filesystem.py").read_text(encoding="utf-8")
    assert '"aadhaar"' not in src and "'aadhaar'" not in src
    assert "uidai" not in src.lower()
