"""API tests: real FastAPI app with sandboxed roots (LLM may fall back offline)."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _make_client(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="lff_api_")
    Path(tmp, "resume_analyzer_notes.pdf").write_text("dummy")
    Path(tmp, "dbms_college_notes.pdf").write_text("dummy")
    monkeypatch.setenv("ALLOWED_ROOTS", tmp)
    monkeypatch.setenv("FILE_INDEX_DB", os.path.join(tmp, "_idx.db"))
    import importlib
    import app.config as cfg

    importlib.reload(cfg)
    import app.tools.filesystem as fs

    importlib.reload(fs)
    fs.build_index(tmp)
    import app.agent.nodes as nodes

    importlib.reload(nodes)
    import app.agent.graph as graph

    graph._graph = None
    importlib.reload(graph)
    import app.main as main

    importlib.reload(main)
    from fastapi.testclient import TestClient

    return TestClient(main.app), tmp


def test_search_returns_real_paths(monkeypatch):
    client, tmp = _make_client(monkeypatch)
    r = client.post("/search", json={"query": "resume analyzer project ki files dhoondo"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"], body
    assert os.path.exists(body["best_match"])
    assert body["best_match"].startswith(tmp)


def test_search_no_results_message(monkeypatch):
    client, tmp = _make_client(monkeypatch)
    r = client.post("/search", json={"query": "aesi file dhoondo jo exist hi nahi karti zzzqxq"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"] == []
    assert body["best_match"] is None
    assert "couldn't find" in (body.get("message") or "").lower()


def test_search_empty_query_rejected(monkeypatch):
    client, tmp = _make_client(monkeypatch)
    r = client.post("/search", json={"query": ""})
    assert r.status_code in (400, 422)


def test_health(monkeypatch):
    client, tmp = _make_client(monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200
    assert "allowed_roots" in r.json()
