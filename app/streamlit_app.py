"""Local File Finder AI — Streamlit presentation layer.

Thin UI over the existing backend. Flow:
    Streamlit -> FastAPI /search -> LangGraph -> LangChain -> Gemma 4
    -> filesystem tools -> ranked real paths.

This file contains NO search/ranking logic; it only renders what the
backend returns (see app/services/api_client.py).

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.api_client import (  # noqa: E402
    BackendHealth,
    SearchOutcome,
    ServiceUnavailable,
    check_health,
    get_api_base_url,
    search_files,
)

st.set_page_config(
    page_title="Local File Finder AI",
    page_icon="🔍",
    layout="centered",
)

# ---------------- minimal, theme-friendly styling ----------------
st.markdown(
    """
    <style>
    .result-card {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 12px;
        padding: 14px 16px;
        margin: 10px 0;
    }
    .best-card {
        border: 1px solid rgba(46, 160, 67, 0.55);
        border-radius: 12px;
        padding: 16px 18px;
        margin: 12px 0;
    }
    .score-badge {
        display: inline-block;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 999px;
        padding: 1px 10px;
        margin-left: 8px;
        white-space: nowrap;
    }
    .reason { opacity: 0.85; font-size: 0.92rem; }
    .filename { font-weight: 600; font-size: 1.02rem; }
    .muted { opacity: 0.7; }
    </style>
    """,
    unsafe_allow_html=True,
)

_DRIVE_RE = re.compile(r"[a-zA-Z]:[\\/]|\\b[a-zA-Z]\\s+drive\\b", re.IGNORECASE)


def _with_location(query: str, location: str) -> str:
    """Fold the optional location box into the natural-language query.

    The backend only accepts `query`, and the query stays primary: if it
    already names a drive, the box is ignored. Otherwise the location is
    appended as plain language ("... in E:\\"), which Gemma understands.
    """
    location = (location or "").strip()
    if not location or _DRIVE_RE.search(query):
        return query
    return f"{query} in {location}"


def _open_path(path: str, folder: bool) -> None:
    """Open a backend-returned path via the local OS. No shell, no commands.

    Only ever called with exact paths from search results, and only if the
    path still exists. Windows-only (os.startfile); elsewhere shows a notice.
    """
    target = os.path.dirname(path) if folder else path
    if not os.path.exists(target):
        st.error("That path no longer exists on disk.")
        return
    try:
        opener = getattr(os, "startfile", None)
        if opener is None:
            st.info("Opening files directly is only supported on Windows.")
            return
        opener(target)
        st.toast("Opened: " + target)
    except Exception:
        st.error("Could not open that location.")


def _run_search_with_phases(query: str) -> SearchOutcome:
    """Run the (slow, CPU-bound LLM) backend call with honest staged status.

    The request runs in a worker thread; the UI polls it and advances the
    stage label as the pipeline progresses. The spinner is real — it lasts
    exactly as long as the backend request does.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(search_files, query)
        status = st.status("🤖 Understanding your request...", expanded=True)
        started = time.time()
        stage = 0
        while not future.done():
            elapsed = time.time() - started
            if elapsed > 45 and stage < 2:
                stage = 2
                status.update(label="🧠 Evaluating the best matches...",
                              state="running")
            elif elapsed > 18 and stage < 1:
                stage = 1
                status.update(label="🔎 Searching your files...",
                              state="running")
            time.sleep(0.5)
        try:
            outcome = future.result()
        finally:
            status.update(label="Done", state="complete")
        return outcome


def _push_history(query: str) -> None:
    history: list[str] = st.session_state.get("history", [])
    history = [q for q in history if q != query]
    history.insert(0, query)
    st.session_state["history"] = history[:8]


# ---------------- sidebar: real backend status ----------------
health: BackendHealth = check_health()

with st.sidebar:
    st.header("Local AI Status")

    def _dot(ok: bool) -> str:
        return "🟢" if ok else "🔴"

    st.write(f"{_dot(health.reachable)} FastAPI")
    st.write(f"{_dot(health.ollama_reachable)} Ollama")
    gemma_ok = health.ollama_reachable and bool(health.model)
    st.write(f"{_dot(gemma_ok)} Gemma 4"
             + (f" `{health.model}`" if health.model else ""))
    st.caption(f"API: {get_api_base_url()}")

    st.divider()
    st.subheader("Search Root")
    roots = health.allowed_roots or ["E:\\"]
    st.code(roots[0], language=None)

    st.divider()
    st.write("🔒 Local processing only")
    st.caption("Queries and file paths never leave this computer.")

    st.divider()
    st.subheader("Recent Searches")
    for i, old in enumerate(st.session_state.get("history", [])):
        if st.button(old, key=f"hist_{i}", use_container_width=True,
                     help="Run this search again"):
            st.session_state["query_input"] = old
            st.session_state["active_query"] = old
            st.session_state["run_search"] = True
            st.rerun()

# ---------------- main screen ----------------
st.title("Local File Finder AI")
st.write("Find files on your computer using natural language. "
         "Your data stays local.")

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.form(key="search_form", clear_on_submit=False):
    query = st.text_input(
        "Search",
        key="query_input",
        placeholder="e.g. mera Aadhaar card E drive mein find karo",
        label_visibility="collapsed",
    )
    col_loc, col_btn = st.columns([3, 1])
    with col_loc:
        location = st.text_input(
            "Search location",
            value=roots[0],
            help="Optional. The natural-language query takes priority — "
                 "this is only used when the query names no location.",
        )
    with col_btn:
        st.write("")  # align with input
        submitted = st.form_submit_button("🔍 Search Files",
                                          use_container_width=True)

if submitted and (query or "").strip():
    st.session_state["active_query"] = query.strip()
    st.session_state["run_search"] = True

# ---------------- run + render ----------------
if st.session_state.get("run_search") and st.session_state.get("active_query"):
    st.session_state["run_search"] = False
    effective = _with_location(st.session_state["active_query"], location)
    try:
        outcome = _run_search_with_phases(effective)
    except ServiceUnavailable as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        st.error("Something went wrong while searching. Please try again.")
        st.stop()
    st.session_state["outcome"] = outcome
    _push_history(st.session_state["active_query"])

outcome: SearchOutcome | None = st.session_state.get("outcome")
if outcome is None:
    st.caption("Try: *mera Aadhaar card E drive mein find karo* · "
               "*college ki DBMS wali files dhoondo* · "
               "*resume analyzer project ki files dhoondo*")
    st.stop()

if not outcome.results:
    st.info("No matching files found.")
    st.caption("Try describing the file differently or specify a drive/folder.")
    if outcome.message:
        st.caption(outcome.message)
    st.stop()

best = outcome.best
st.subheader("🎯 Best Match")
st.markdown(
    f'<div class="best-card">'
    f'<div class="filename">{best.filename}'
    f'<span class="score-badge">{best.score * 100:.0f}% match</span></div>'
    f"</div>",
    unsafe_allow_html=True,
)
st.code(best.path, language=None)
st.markdown(f'<div class="reason">**Why this matched:** {best.reason}</div>',
            unsafe_allow_html=True)
b1, b2 = st.columns(2)
with b1:
    if st.button("📂 Open Folder", key="best_folder"):
        _open_path(best.path, folder=True)
with b2:
    if st.button("📄 Open File", key="best_file"):
        _open_path(best.path, folder=False)

others = [r for r in outcome.results if r.path != best.path]
if others:
    st.subheader("Other likely matches")
    for idx, r in enumerate(others, start=2):
        st.markdown(
            f'<div class="result-card">'
            f'<div class="filename">{idx}. {r.filename}'
            f'<span class="score-badge">{r.score * 100:.0f}%</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.code(r.path, language=None)
        st.markdown(f'<div class="reason">{r.reason}</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📂 Open Folder", key=f"folder_{idx}"):
                _open_path(r.path, folder=True)
        with c2:
            if st.button("📄 Open File", key=f"folder_file_{idx}"):
                _open_path(r.path, folder=False)

st.caption(f"Showing {len(outcome.results)} result(s) for: *{outcome.query}*")
