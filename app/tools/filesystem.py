"""Read-only local filesystem tools with path security and a small SQLite index.

Design:
- Gemma (LLM) is the brain: it interprets messy natural language.
- These Python functions are the hands: they touch the real Windows filesystem.
- No hardcoded domain synonyms (no "aadhar -> uid" maps). Matching is generic:
  token overlap + fuzzy similarity over filename / directory names / extension / metadata.
- SQLite index (local file) avoids re-walking the whole drive on every query.
  Live walk is used as a fallback / refresh path. Both are read-only.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from difflib import SequenceMatcher
from pathlib import Path

from langchain_core.tools import tool

from app.config import settings

# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------


class PathSecurityError(ValueError):
    pass


def _normalise_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in settings.allowed_roots:
        try:
            p = Path(raw).resolve()
        except Exception:
            p = Path(raw).absolute()
        roots.append(p)
    return roots


def _is_within(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_and_validate(raw_path: str) -> Path:
    """Resolve a user/LLM-supplied path and ensure it sits inside ALLOWED_ROOTS.

    Protects against: ../ traversal, absolute paths outside roots,
    malformed Windows paths, and symlink/junction escapes (via resolve()).
    """
    if not raw_path or not raw_path.strip():
        raise PathSecurityError("Empty path is not allowed.")
    candidate = raw_path.strip().strip('"').strip("'")
    # Reject obviously malformed inputs early.
    if "\x00" in candidate:
        raise PathSecurityError("Invalid path (null byte).")
    try:
        resolved = Path(candidate).resolve()
    except Exception as exc:
        raise PathSecurityError(f"Malformed path: {candidate!r}") from exc

    for root in _normalise_roots():
        # Compare case-insensitively on Windows (C: vs c:, E:\ vs e:\).
        try:
            if _is_within(resolved, root) or os.path.normcase(
                str(resolved)
            ) == os.path.normcase(str(root)):
                return resolved
            # Also compare normcase string prefix as a belt-and-braces check.
            r = os.path.normcase(str(root))
            c = os.path.normcase(str(resolved))
            if c == r or c.startswith(r.rstrip(os.sep) + os.sep):
                return resolved
        except Exception:
            continue
    raise PathSecurityError(
        f"Access denied: {candidate!r} is outside ALLOWED_ROOTS={settings.allowed_roots}"
    )


# ---------------------------------------------------------------------------
# Generic text helpers (NOT domain synonym lists)
# ---------------------------------------------------------------------------

# Generic stopwords only (language filler, not domain knowledge).
_STOPWORDS = frozenset(
    {
        "mera", "meri", "mere", "mein", "me", "main", "ko", "ki", "ka", "ke",
        "kaha", "kahaan", "pada", "hai", "ho", "woh", "wo", "jo", "dhoondo",
        "dhoondho", "find", "karo", "kar", "wal", "wali", "wale", "wala",
        "the", "a", "an", "my", "please", "karo", "mujhe", "batao", "chahiye",
        "file", "files", "folder", "drive", "laptop", "computer", "mein",
        "e", "c", "d", "me", "do", "show", "search", "locate", "where",
        "aesi", "aisi", "jo", "exist", "karti", "karte", "nahi", "nahin",
        "hi", "hello", "hey", "namaste",
    }
)

_EXT_RE = re.compile(r"\b([a-z0-9]{2,5})\b", re.IGNORECASE)
_KNOWN_EXTENSIONS = frozenset(
    {
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "md",
        "csv", "jpg", "jpeg", "png", "gif", "bmp", "mp4", "mp3", "wav",
        "zip", "rar", "7z", "py", "js", "ts", "java", "c", "cpp", "json",
        "html", "css", "exe", "apk",
    }
)

# Generic dependency/cache dirs skipped while walking: they hold thousands of
# third-party files (never the user's own documents) and would exhaust the walk
# budget before user content is reached. Domain-agnostic hygiene, NOT synonyms.
SKIP_DIRS = frozenset(
    {
        "__pycache__", ".git", ".hg", ".svn", "node_modules", "site-packages",
        "dist-info", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".tox",
        ".ipynb_checkpoints",
    }
)


def tokenize(text: str) -> list[str]:
    """Generic tokenizer: lowercase alphanumeric words, minus filler words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = [w for w in words if w not in _STOPWORDS and len(w) >= 2]
    # Keep single-letter drive hints out; keep everything else generic.
    return out


def detect_extension_hint(text: str) -> str | None:
    """Detect an extension hint generically (e.g. user typed 'pdfs' -> 'pdf').

    This is NOT a synonym dictionary: it only normalises a literal extension
    word the user actually typed.
    """
    lowered = text.lower()
    for m in _EXT_RE.finditer(lowered):
        word = m.group(1).lower().rstrip("s")  # crude plural handling: pdfs->pdf
        if word in _KNOWN_EXTENSIONS:
            # Only accept if the user mentioned it near a file-ish word OR at all?
            # Milestone 1: accept any explicit extension mention.
            return word
    # Handle explicit dotted form: ".pdf"
    dotted = re.search(r"\.([a-z0-9]{2,5})\b", lowered)
    if dotted and dotted.group(1) in _KNOWN_EXTENSIONS:
        return dotted.group(1)
    return None


def _fuzzy(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def score_candidate(
    query_tokens: list[str], file_path: str, extension_filter: str | None = None
) -> tuple[float, str]:
    """Generic relevance score in [0,1] + short reason.

    Signals (no domain knowledge):
    - token overlap with filename stem
    - token overlap with parent directory names
    - fuzzy similarity for transliteration robustness (aadhar vs aadhaar)
    - extension agreement when a filter/hint exists
    """
    p = Path(file_path)
    stem = p.stem.lower()
    parent_names = " ".join(part.lower() for part in p.parts[-4:-1])
    full_name = p.name.lower()
    ext = p.suffix.lower().lstrip(".")

    if not query_tokens:
        return 0.0, "no usable search terms"

    stem_tokens = set(re.findall(r"[a-z0-9]+", stem))
    parent_tokens = set(re.findall(r"[a-z0-9]+", parent_names))
    name_tokens = set(re.findall(r"[a-z0-9]+", full_name))

    exact = sum(1 for t in query_tokens if t in stem_tokens)
    partial = 0.0
    for qt in query_tokens:
        if any((qt in st or st in qt) and min(len(qt), len(st)) >= 5
               for st in stem_tokens if len(st) >= 3):
            partial += 0.6
        # Fuzzy: handles spelling/transliteration variants generically.
        best = max((_fuzzy(qt, st) for st in stem_tokens), default=0.0)
        if best >= 0.82:
            partial += 0.5 * best
    dir_hits = sum(1 for t in query_tokens if t in parent_tokens or t in name_tokens)

    base = (exact * 1.0 + partial * 0.7 + dir_hits * 0.35) / max(len(query_tokens), 1)

    # Extension agreement boost (generic, only when user asked for a type).
    ext_reason = ""
    if extension_filter:
        if ext == extension_filter.lower():
            base += 0.35
            ext_reason = f", extension .{ext} matches request"
        else:
            base -= 0.25

    score = max(0.0, min(1.0, base / 1.6))
    why = (
        f"filename {exact}/{len(query_tokens)} exact token hits"
        f"{ext_reason}; dir-context hits={dir_hits}"
    )
    return round(score, 3), why


# ---------------------------------------------------------------------------
# SQLite index (local, simple — no embeddings in milestone 1)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    parent TEXT NOT NULL,
    extension TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    mtime REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_files_filename ON files(filename);
CREATE INDEX IF NOT EXISTS idx_files_parent ON files(parent);
CREATE INDEX IF NOT EXISTS idx_files_ext ON files(extension);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def _connect() -> sqlite3.Connection:
    db = Path(settings.index_db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(_SCHEMA)
    return conn


def build_index(root: str | Path, max_files: int | None = None) -> dict:
    """Walk one allowed root and (re)populate the SQLite index. Read-only w.r.t. user files."""
    validated = resolve_and_validate(str(root))
    if not validated.exists():
        raise FileNotFoundError(f"Location does not exist: {validated}")
    if not validated.is_dir():
        raise NotADirectoryError(f"Not a directory: {validated}")

    started = time.time()
    conn = _connect()
    visited = 0
    indexed = 0
    try:
        conn.execute("DELETE FROM files WHERE path LIKE ?", (str(validated) + "%",))
        batch: list[tuple] = []
        for dirpath, _dirnames, filenames in os.walk(validated, onerror=None, followlinks=False):
            # Stay inside the validated root even if junctions try to escape.
            try:
                cur = Path(dirpath).resolve()
            except Exception:
                continue
            if cur != validated and not _is_within(cur, validated):
                _dirnames[:] = []
                continue
            _dirnames[:] = [d for d in _dirnames
                            if d not in SKIP_DIRS and not d.startswith("$")]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                visited += 1
                if max_files and visited > max_files:
                    break
                try:
                    st = os.stat(full, follow_symlinks=False)
                    if os.path.islink(full):
                        # Skip symlinks that escape the root.
                        try:
                            if not _is_within(Path(full).resolve(), validated):
                                continue
                        except Exception:
                            continue
                    ext = Path(fn).suffix.lower().lstrip(".")
                    batch.append((full, fn.lower(), os.path.dirname(full).lower(), ext, st.st_size, st.st_mtime))
                    indexed += 1
                except (OSError, PermissionError):
                    continue
                if len(batch) >= 2000:
                    conn.executemany(
                        "INSERT OR REPLACE INTO files(path,filename,parent,extension,size,mtime) VALUES(?,?,?,?,?,?)",
                        batch,
                    )
                    batch.clear()
            if max_files and visited > max_files:
                break
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO files(path,filename,parent,extension,size,mtime) VALUES(?,?,?,?,?,?)",
                batch,
            )
        conn.execute(
            "INSERT OR REPLACE INTO meta(k,v) VALUES('last_index_root',?)", (str(validated),)
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(k,v) VALUES('last_index_ts',?)", (str(time.time()),)
        )
        conn.commit()
    finally:
        conn.close()
    return {"root": str(validated), "visited": visited, "indexed": indexed,
            "seconds": round(time.time() - started, 2)}


def _prune(dirpath: str, dirnames: list[str], root: Path) -> None:
    """In-place prune: escaping symlinks + generic dependency/cache dirs."""
    kept: list[str] = []
    for d in dirnames:
        if d in SKIP_DIRS or d.startswith("$"):
            continue
        full_d = os.path.join(dirpath, d)
        try:
            if os.path.islink(full_d) and not _is_within(Path(full_d).resolve(), root):
                continue
        except Exception:
            continue
        kept.append(d)
    dirnames[:] = sorted(kept)


def _score_file(full: str, fn: str, query_tokens: list[str],
                extension_filter: str | None) -> dict | None:
    if extension_filter and not fn.lower().endswith("." + extension_filter.lower()):
        return None
    try:
        if os.path.islink(full):
            return None  # file-level symlinks skipped (dir escapes handled in _prune)
        st = os.stat(full, follow_symlinks=False)
    except (OSError, PermissionError):
        return None
    score, why = score_candidate(query_tokens, full, extension_filter)
    if score <= 0.0:
        return None
    return {"path": full, "filename": fn,
            "extension": Path(fn).suffix.lower().lstrip("."),
            "size": st.st_size, "modified": st.st_mtime,
            "score": score, "reason": why}


def _live_walk(root: Path, query_tokens: list[str], extension_filter: str | None,
               limit: int, deadline: float) -> list[dict]:
    """Bounded live walk. Never raises on permission errors.

    Fairness design: huge roots (venvs, game installs) must not starve normal
    folders. Root-level files are scanned first, then each top-level directory
    gets its own bounded walk (round-robin), so every folder gets coverage.
    """
    scored: list[tuple[float, dict]] = []
    seen: set[str] = set()

    def _add(hit: dict | None) -> None:
        if hit and hit["path"] not in seen:
            seen.add(hit["path"])
            scored.append((hit["score"], hit))

    # Phase 0: files directly in root (fast, catches drive-root documents).
    try:
        with os.scandir(root) as it:
            entries = sorted(list(it), key=lambda e: e.name.lower())
    except (OSError, PermissionError):
        entries = []
    topdirs: list[str] = []
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False):
                _add(_score_file(entry.path, entry.name, query_tokens, extension_filter))
            elif entry.is_dir(follow_symlinks=False):
                topdirs.append(entry.path)
        except (OSError, PermissionError):
            continue
        if time.time() > deadline:
            break

    # Phase 1: round-robin — each top-level dir gets a bounded walk.
    total_file_budget = settings.max_files_visited
    per_dir_files = max(2000, total_file_budget // max(len(topdirs), 1)) if topdirs else 0
    per_dir_dirs = max(500, settings.max_dirs_per_search // max(len(topdirs), 1)) if topdirs else 0
    for top in topdirs:
        if time.time() > deadline:
            break
        try:
            if os.path.islink(top) and not _is_within(Path(top).resolve(), root):
                continue
        except Exception:
            continue
        files_here = 0
        dirs_here = 0
        try:
            walker = os.walk(top, onerror=None, followlinks=False)
        except Exception:
            continue
        for dirpath, dirnames, filenames in walker:
            if time.time() > deadline:
                break
            dirs_here += 1
            if dirs_here > per_dir_dirs:
                break
            _prune(dirpath, dirnames, root)
            for fn in filenames:
                if time.time() > deadline:
                    break
                files_here += 1
                if files_here > per_dir_files:
                    break
                _add(_score_file(os.path.join(dirpath, fn), fn,
                                 query_tokens, extension_filter))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:limit]]


def _search_index(root: Path, query_tokens: list[str], extension_filter: str | None,
                  limit: int) -> list[dict]:
    """LIKE-based lookup over the local SQLite index, then generic re-scoring."""
    if not Path(settings.index_db_path).exists():
        return []
    conn = sqlite3.connect(str(settings.index_db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Pull a bounded candidate set with OR-LIKE, then score precisely in Python.
        clauses: list[str] = ["path LIKE ?"]
        params: list = [str(root) + "%"]
        if extension_filter:
            clauses.append("extension = ?")
            params.append(extension_filter.lower())
        token_clauses: list[str] = []
        token_params: list = []
        for t in query_tokens[:6]:
            token_clauses.append("(filename LIKE ? OR parent LIKE ?)")
            token_params += [f"%{t}%", f"%{t}%"]
        sql = "SELECT * FROM files WHERE " + " AND ".join(clauses)
        if token_clauses:
            sql += " AND (" + " OR ".join(token_clauses) + ")"
        sql += " LIMIT 400"
        try:
            rows = conn.execute(sql, params + token_params).fetchall()
        except Exception:
            return []
        scored: list[tuple[float, dict]] = []
        for r in rows:
            score, why = score_candidate(query_tokens, r["path"], extension_filter)
            if score <= 0.0:
                continue
            # Verify the file still exists (index may be stale); skip ghosts.
            if not os.path.exists(r["path"]):
                continue
            scored.append((score, {
                "path": r["path"], "filename": Path(r["path"]).name,
                "extension": r["extension"], "size": r["size"],
                "modified": r["mtime"], "score": score, "reason": why,
            }))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:limit]]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Core (plain Python — understandable hands)
# ---------------------------------------------------------------------------

def search_files_core(location: str, query: str, limit: int = 10,
                      extension_filter: str | None = None) -> list[dict]:
    """Primary search: validate -> index lookup -> bounded live walk -> generic ranking."""
    from app.schemas.search import SearchFilesArgs

    args = SearchFilesArgs(location=location, query=query, limit=limit,
                           extension_filter=extension_filter)
    root = resolve_and_validate(args.location)
    if not root.exists():
        raise FileNotFoundError(f"Location does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    query_tokens = tokenize(args.query)
    ext = (args.extension_filter or detect_extension_hint(args.query) or None)
    if ext:
        ext = ext.lower().lstrip(".")
    # Noise gate: content queries must clear a minimum score; pure
    # "browse this type" queries (no content tokens, only an extension hint)
    # are exempt since their fixed score is intentionally modest.
    min_score = 0.30 if query_tokens else 0.0
    deadline = time.time() + settings.search_timeout_seconds

    # If the query had no usable tokens but has an extension hint (e.g. "PDFs dhoondo"),
    # list recent files of that type instead of returning nothing.
    if not query_tokens and not ext:
        raise ValueError("Could not derive any search terms from the query.")
    if not query_tokens and ext:
        query_tokens = []

    indexed = [c for c in _search_index(root, query_tokens, ext, args.limit)
               if c["score"] >= min_score]
    if indexed and len(indexed) >= 3:
        return indexed[: args.limit]

    live = [c for c in _live_walk(root, query_tokens, ext, args.limit, deadline)
            if c["score"] >= min_score]
    if live:
        # Merge: prefer live results, keep index-only extras.
        seen = {c["path"] for c in live}
        for c in indexed:
            if c["path"] not in seen and len(live) < args.limit:
                live.append(c)
                seen.add(c["path"])
        live.sort(key=lambda c: c["score"], reverse=True)
        return live[: args.limit]

    # Extension-only fallback: newest files of that type.
    if ext and not query_tokens:
        conn = sqlite3.connect(str(settings.index_db_path)) if Path(settings.index_db_path).exists() else None
        if conn is not None:
            try:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM files WHERE path LIKE ? AND extension = ? ORDER BY mtime DESC LIMIT ?",
                    (str(root) + "%", ext, args.limit),
                ).fetchall()
                return [{
                    "path": r["path"], "filename": Path(r["path"]).name,
                    "extension": r["extension"], "size": r["size"], "modified": r["mtime"],
                    "score": 0.5, "reason": f"recent .{ext} file in requested location",
                } for r in rows if os.path.exists(r["path"])]
            finally:
                conn.close()
    return indexed[: args.limit] if indexed else []


def list_directory_core(location: str, limit: int = 100) -> list[dict]:
    from app.schemas.search import ListDirectoryArgs

    args = ListDirectoryArgs(location=location, limit=limit)
    root = resolve_and_validate(args.location)
    if not root.exists():
        raise FileNotFoundError(f"Location does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    out: list[dict] = []
    try:
        with os.scandir(root) as it:
            for entry in it:
                try:
                    st = entry.stat(follow_symlinks=False)
                    out.append({
                        "name": entry.name, "path": str(Path(entry.path)),
                        "is_dir": entry.is_dir(follow_symlinks=False),
                        "size": st.st_size, "modified": st.st_mtime,
                    })
                except (OSError, PermissionError):
                    out.append({"name": entry.name, "path": str(Path(entry.path)),
                                "is_dir": None, "size": 0, "modified": 0.0,
                                "note": "inaccessible"})
                if len(out) >= args.limit:
                    break
    except PermissionError as exc:
        raise PermissionError(f"Permission denied: {root}") from exc
    return out


def get_file_info_core(path: str) -> dict:
    from app.schemas.search import GetFileInfoArgs

    args = GetFileInfoArgs(path=path)
    target = resolve_and_validate(args.path)
    if not os.path.lexists(target):
        raise FileNotFoundError(f"File does not exist: {target}")
    st = os.stat(target, follow_symlinks=False)
    return {
        "path": str(target), "filename": target.name,
        "extension": target.suffix.lower().lstrip("."),
        "size": st.st_size, "modified": st.st_mtime,
        "is_dir": target.is_dir(), "is_file": target.is_file(),
    }


# ---------------------------------------------------------------------------
# LangChain tool wrappers (the controlled hands Gemma may call)
# ---------------------------------------------------------------------------

@tool
def search_files(location: str, query: str, limit: int = 10,
                 extension_filter: str | None = None) -> list:
    """Search the local Windows filesystem (read-only) under an allowed root.

    Args:
        location: directory to search, e.g. 'E:\\'.
        query: natural-language search terms, e.g. 'aadhar card'.
        limit: max candidates to return (1-50).
        extension_filter: optional lowercase extension without dot, e.g. 'pdf'.
    Returns a list of candidate file dicts. Paths always come from the real filesystem.
    """
    return search_files_core(location, query, limit, extension_filter)


@tool
def list_directory(location: str, limit: int = 100) -> list:
    """List a directory's immediate children (read-only). Location must be inside ALLOWED_ROOTS."""
    return list_directory_core(location, limit)


@tool
def get_file_info(path: str) -> dict:
    """Return metadata for one file (read-only). Path must be inside ALLOWED_ROOTS."""
    return get_file_info_core(path)


FILESYSTEM_TOOLS = [search_files, list_directory, get_file_info]
