"""Security: paths outside ALLOWED_ROOTS must be refused."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _sandbox(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="lff_sec_")
    monkeypatch.setenv("ALLOWED_ROOTS", tmp)
    monkeypatch.setenv("FILE_INDEX_DB", os.path.join(tmp, "_idx.db"))
    import importlib
    import app.config as cfg

    importlib.reload(cfg)
    import app.tools.filesystem as fs

    importlib.reload(fs)
    return tmp, fs


def test_absolute_escape_blocked(monkeypatch):
    tmp, fs = _sandbox(monkeypatch)
    with pytest.raises(Exception):
        fs.resolve_and_validate("C:\\Windows\\System32")


def test_traversal_blocked(monkeypatch):
    tmp, fs = _sandbox(monkeypatch)
    evil = os.path.join(tmp, "..", "..", "Windows")
    with pytest.raises(Exception):
        fs.resolve_and_validate(evil)


def test_search_outside_root_blocked(monkeypatch):
    tmp, fs = _sandbox(monkeypatch)
    with pytest.raises(Exception):
        fs.search_files_core("C:\\Windows", "drivers", limit=5)


def test_get_file_info_outside_blocked(monkeypatch):
    tmp, fs = _sandbox(monkeypatch)
    with pytest.raises(Exception):
        fs.get_file_info_core("C:\\Windows\\explorer.exe")


def test_tools_are_read_only():
    src = Path(ROOT, "app", "tools", "filesystem.py").read_text(encoding="utf-8")
    for dangerous in ["os.remove", "os.unlink", "shutil.rmtree", "os.rename",
                      "shutil.move", "os.system", "subprocess", "os.exec"]:
        assert dangerous not in src, f"read-only violation: {dangerous} found"
