"""CodeLens - Git-Aware Multi-Stage Code Retrieval Engine.
Streamlit Web Application & Interactive Dashboard.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import difflib
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from codetrace.chunker import chunk_source_code
from codetrace.evaluate import DEFAULT_BENCHMARK_QUERIES, evaluate_pipeline
from codetrace.git_tracker import GitTracker
from codetrace.intent import analyze_query
from codetrace.pipeline import Pipeline
from codetrace.versions import HistoryIndex, build_history, link_versions, save_sqlite


def inject_app_css() -> None:
    """Apply a polished dashboard theme to the app."""
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
            background: linear-gradient(180deg, #f3f7ff 0%, #edf3ff 26%, #eef2f8 100%);
            color: #142033;
        }
        [data-testid="stHeader"] {
            height: 2.25rem;
            min-height: 2.25rem !important;
            background: transparent;
        }
        [data-testid="stHeader"] [data-testid="stToolbar"] {
            height: 2.25rem;
            min-height: 2.25rem;
        }
        .stApp * {
            color: #142033 !important;
        }
        .block-container {
            padding-top: 0.45rem;
            padding-bottom: 1.5rem;
        }
        section[data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, rgba(15,25,38,0.96), rgba(18,30,48,0.94));
            border-right: 1px solid rgba(134, 167, 255, 0.2);
        }
        section[data-testid="stSidebar"] [data-testid="stLogoSpacer"] {
            height: 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            margin-bottom: 0 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #edf5ff !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(255,255,255,0.72);
            border-radius: 14px;
            padding: 0.35rem;
            border: 1px solid rgba(125, 150, 197, 0.24);
            box-shadow: 0 10px 28px rgba(34, 52, 84, 0.08);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #22314d;
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-list"] [aria-selected="true"] {
            background: linear-gradient(135deg, #1a73e8, #6d5ef3);
            color: white !important;
            box-shadow: 0 12px 24px rgba(55, 99, 255, 0.2);
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(72, 105, 173, 0.18);
            background: linear-gradient(135deg, #0f4aa8, #4e6ce3);
            color: white !important;
            font-weight: 700;
            transition: all 0.2s ease;
            box-shadow: 0 10px 26px rgba(71, 109, 229, 0.24);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(71, 109, 229, 0.28);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1d74f5, #7369f0);
            border: none;
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div,
        .stTextInput > div > div,
        .stTextArea > div > div,
        .stSelectbox > div > div,
        .stNumberInput > div > div {
            background: #ffffff !important;
            border: 1px solid rgba(94, 112, 150, 0.28) !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 12px rgba(34, 52, 84, 0.04) !important;
        }
        input, textarea, [data-baseweb="select"] input, [data-baseweb="input"] input {
            color: #111827 !important;
            background: #ffffff !important;
            -webkit-text-fill-color: #111827 !important;
            caret-color: #111827 !important;
        }
        .block-container .stAlert {
            border-radius: 14px;
        }
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(90, 110, 160, 0.14);
        }
        .stMetric {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(120, 145, 201, 0.18);
            border-radius: 14px;
            padding: 0.7rem 0.8rem;
            box-shadow: 0 8px 18px rgba(41, 57, 84, 0.05);
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(17, 42, 75, 0.97), rgba(38, 31, 64, 0.92));
            border: 1px solid rgba(141, 171, 255, 0.18);
            border-radius: 14px;
            padding: 0.7rem 1rem;
            box-shadow: 0 18px 35px rgba(23, 37, 62, 0.18);
            margin-bottom: 0.55rem;
        }
        .hero-card * {
            color: #f1f7ff !important;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin-bottom: 0.1rem;
            background: linear-gradient(90deg, #d9eeff, #a7d9ff, #d7c6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub {
            color: #edf6ff;
            font-size: 0.92rem;
            opacity: 0.93;
        }
        .status-chip {
            display: inline-block;
            background: rgba(84, 206, 176, 0.16);
            color: #0d7d63;
            border: 1px solid rgba(18, 128, 108, 0.2);
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.8rem;
            font-weight: 800;
            margin-top: 0.15rem;
        }
        .stSidebar .stSelectbox label,
        .stSidebar .stTextInput label,
        .stSidebar .stNumberInput label,
        .stSidebar .stButton p,
        .stSidebar .stMarkdown p,
        .stSidebar .stMarkdown li,
        .stSidebar .stCheckbox label {
            color: #edf5ff !important;
        }
        .stMain .stMarkdown p,
        .stMain .stMarkdown li,
        .stMain .stMarkdown h1,
        .stMain .stMarkdown h2,
        .stMain .stMarkdown h3,
        .stMain .stMarkdown h4,
        .stMain .stDataFrame,
        .stMain .stDataFrame * {
            color: #142033 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_demo_git_repo(dest_dir: str):
    """Create a real local Git repository with commit history for the demo."""
    import git

    repo_path = Path(dest_dir)
    if repo_path.exists():
        try:
            existing_repo = git.Repo(str(repo_path))
            for existing_commit in existing_repo.iter_commits():
                for existing_object in existing_commit.tree.traverse():
                    if existing_object.type == "blob":
                        existing_object.data_stream.read()
            return str(repo_path)
        except Exception:
            recovery_index = 1
            while True:
                recovered_path = repo_path.with_name(f"{repo_path.name}_recovered_{recovery_index}")
                if not recovered_path.exists():
                    return create_demo_git_repo(str(recovered_path))
                try:
                    recovered_repo = git.Repo(str(recovered_path))
                    for recovered_commit in recovered_repo.iter_commits():
                        for recovered_object in recovered_commit.tree.traverse():
                            if recovered_object.type == "blob":
                                recovered_object.data_stream.read()
                    return str(recovered_path)
                except Exception:
                    recovery_index += 1

    repo_path.mkdir(parents=True, exist_ok=True)

    repo = git.Repo.init(str(repo_path))

    src_dir = repo_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    file_preprocess = src_dir / "preprocess.py"
    file_preprocess.write_text(
        '''def clean_data(raw_items):
    """Strip whitespace and remove empty strings from raw inputs."""
    return [x.strip() for x in raw_items if x]

def normalize_input(raw_input):
    """Normalize raw input items before passing to the main processing engine."""
    cleaned = clean_data(raw_input)
    return [item.lower() for item in cleaned]
''',
        encoding="utf-8",
    )

    file_auth = src_dir / "auth.py"
    file_auth.write_text(
        '''def verify_token(token):
    """Verify security bearer token."""
    return token.startswith("bearer_") and len(token) > 12

def validate_user(username, password):
    """Validate user credentials before authentication."""
    return len(username) >= 3 and len(password) >= 8

def authenticate_user(credentials):
    """Authenticate user credentials and verify access token."""
    if not validate_user(credentials.get("user", ""), credentials.get("pass", "")):
        return False
    return verify_token(credentials.get("token", ""))
''',
        encoding="utf-8",
    )

    file_main = src_dir / "main.py"
    file_main.write_text(
        '''from preprocess import normalize_input
from auth import authenticate_user

def process_pipeline(batch):
    """Process incoming data batch through normalization."""
    normalized = normalize_input(batch)
    return {"status": "ok", "items": normalized}

def main():
    print("CodeLens Core Engine Running")
''',
        encoding="utf-8",
    )

    repo.index.add([str(file_preprocess), str(file_auth), str(file_main)])
    author = git.Actor("Dev Lead", "dev@codelens.io")
    repo.index.commit("Initial commit: auth, preprocess, and main pipeline", author=author, committer=author)

    file_session = src_dir / "session.py"
    file_session.write_text(
        '''def create_session(user_id):
    """Initialize user session immediately after successful authentication."""
    return {"session_id": f"sess_{user_id}", "created_at": "2026-10-02", "active": True}

def terminate_session(session_id):
    """Invalidate active session."""
    return {"session_id": session_id, "active": False}
''',
        encoding="utf-8",
    )
    repo.index.add([str(file_session)])
    repo.index.commit("Add session management module", author=author, committer=author)

    return str(repo_path)


def ensure_repository_input(repo_input: str) -> str:
    """Create the default demo repo when the configured path is missing or invalid."""
    requested = (repo_input or "").strip()
    if not requested or requested in {".", str(Path.cwd()), str(Path.cwd().resolve())}:
        requested = "data/demo_repo"

    default_demo = Path("data/demo_repo").resolve()
    repo_path = Path(requested).expanduser()

    if repo_path.exists() and repo_path.is_dir():
        resolved = repo_path.resolve()
        if resolved == default_demo or repo_path.name == "demo_repo":
            return create_demo_git_repo(str(resolved))
        if (resolved / ".git").exists():
            return str(resolved)

        # If the user is pointing at the current workspace root or another non-git folder,
        # fall back to the demo repository instead of indexing the wrong local folder.
        if resolved == Path.cwd().resolve():
            return create_demo_git_repo(str(default_demo))

    if repo_path == default_demo or repo_path.name == "demo_repo" or (requested.endswith("demo_repo") and not requested.startswith("http")):
        return create_demo_git_repo(str(default_demo))

    if not repo_path.is_absolute():
        fallback = (Path.cwd() / repo_path).resolve()
        if fallback.exists() and fallback.is_dir() and (fallback / ".git").exists():
            return str(fallback)

    if not repo_path.exists() or not repo_path.is_dir():
        return create_demo_git_repo(str(default_demo))

    return str(repo_path.resolve())


def main():
    # Page Configuration
    st.set_page_config(
        page_title="CodeLens — Git-Aware Code Retrieval",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_app_css()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🔎 CodeLens</div>
            <div class="hero-sub">Git-aware multi-stage code retrieval, intelligent version tracking, and benchmark-grade code search.</div>
            <div class="status-chip">Live repository intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    MODELS = {
        "MiniLM (fast, CPU)": "sentence-transformers/all-MiniLM-L6-v2",
        "Jina code (better, slower)": "jinaai/jina-embeddings-v2-base-code",
    }

    MODE_LABELS = {
        "dense": "A · Dense Embedding only (baseline)",
        "hybrid": "B · Hybrid BM25 + Dense + RRF",
        "rerank": "C · Hybrid + Intent / Structural Rerank",
        "full": "D · Full Pipeline (+ Confidence Second Pass)",
    }

    INTENT_COLORS = {
        "DATA_FLOW": "orange",
        "FUNCTION_LOOKUP": "blue",
        "CALL_FLOW": "green",
        "IMPLEMENTATION": "violet",
        "VERSION_CHANGE": "red",
        "GENERAL_CODE_SEARCH": "gray",
    }

    S = st.session_state
    tracker = GitTracker()

    # ------------------------------------------------------------------ Search & Ranking
    def run_search(query: str, mode: str, k: int = 10, commit_sha: Optional[str] = None):
        t0 = time.perf_counter()
        pool = 50
        res = S.pipe.search(query, mode=mode, k=pool, pool=pool)

        # Filter by commit presence if requested
        if commit_sha is not None and commit_sha != "ALL":
            commit_indices = [i for i, c in enumerate(S.hist.commits) if c["sha"] == commit_sha]
            if commit_indices:
                c_idx = commit_indices[0]
                res = [r for r in res if c_idx in S.hist.presence[S.hist.index_of.get(r.chunk.id, -1)]]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return res[:k], elapsed_ms

    def commit_label(hist: HistoryIndex, i: int) -> str:
        m = hist.commits[i]
        msg = m["message"][:45] + "..." if len(m["message"]) > 45 else m["message"]
        return f"#{i} · {m['sha'][:7]} · {m['date']} · {msg}"

    # ------------------------------------------------------------------ UI Components
    def render_insight_panel(chunk, rank: int):
        """Render CodeLens Insight Panel: Relationships, Call Flow, and Evolution."""
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("#### 🔗 Relationships & Graph Neighborhood")
            neighborhood = S.pipe.code_graph.get_neighborhood(chunk.id)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Calls (Callees):**")
                if neighborhood["calls"]:
                    for fn in neighborhood["calls"]:
                        st.markdown(f"- `{fn}()`")
                else:
                    st.caption("No outgoing calls")

            with col_b:
                st.markdown("**Called by (Callers):**")
                if neighborhood["called_by"]:
                    for fn in neighborhood["called_by"]:
                        st.markdown(f"- `{fn}()`")
                else:
                    st.caption("No incoming callers detected")

            if chunk.parent_class:
                st.markdown(f"**Parent Class:** `{chunk.parent_class}`")
            if chunk.imports:
                st.markdown(f"**Imports:** {', '.join([f'`{imp}`' for imp in chunk.imports[:5]])}")

        with c2:
            st.markdown("#### 🕰️ Function Evolution & Lineage")
            lineage = S.hist.get_symbol_lineage(chunk.name)
            if len(lineage) <= 1:
                st.caption(f"Single version detected (first seen in commit `{chunk.commit}`).")
            else:
                st.markdown(f"**{len(lineage)} versions tracked across commits:**")
                ver_options = [f"Commit {v['sha'][:7]} ({v['date']})" for v in lineage]
                sel_ver = st.selectbox(
                    "Compare version diff",
                    range(len(lineage)),
                    format_func=lambda i: ver_options[i],
                    key=f"lineage_sel_{rank}_{chunk.id[:8]}",
                )
                v_curr = lineage[sel_ver]
                if sel_ver > 0:
                    v_prev = lineage[sel_ver - 1]
                    diff_text = "\n".join(
                        difflib.unified_diff(
                            v_prev["code"].splitlines(),
                            v_curr["code"].splitlines(),
                            f"Commit {v_prev['sha'][:7]}",
                            f"Commit {v_curr['sha'][:7]}",
                            lineterm="",
                        )
                    )
                    st.code(diff_text or "(No textual code difference)", language="diff")
                else:
                    st.code(v_curr["code"], language=chunk.language)

    def show_results(results, key_prefix: str = "res"):
        hist = S.hist
        for rank, r in enumerate(results, 1):
            c = r.chunk
            idx = hist.index_of.get(c.id, 0)
            pres = hist.presence[idx] if idx < len(hist.presence) else [0]
            first_commit = hist.commits[pres[0]]
            last_commit = hist.commits[pres[-1]]

            score_display = f"{r.score:.3f}"
            header = f"#{rank}  `{c.name or 'code_block'}`  ·  {c.path}:{c.start}–{c.end}  ·  Score: **{score_display}**"

            with st.expander(header, expanded=rank == 1):
                col_meta1, col_meta2 = st.columns([2, 1])
                with col_meta1:
                    st.caption(
                        f"**Language:** `{c.language}` · **Commit:** `{c.commit}` · "
                        f"**Active in:** {len(pres)}/{len(hist.commits)} commits "
                        f"(from `{first_commit['sha'][:7]}` to `{last_commit['sha'][:7]}`)"
                    )
                with col_meta2:
                    if r.evidence.get("second_pass"):
                        st.badge("Confidence 2nd Pass Triggered", color="violet")

                # Code display
                st.code(c.code, language=c.language)

                # Retrieval Evidence Table
                st.markdown("**🔬 Retrieval Evidence & Signal Breakdown**")
                ev = {k: v for k, v in r.evidence.items() if v is not None}
                st.dataframe(pd.DataFrame([ev]), hide_index=True, use_container_width=True)

                # CodeLens Insight Panel
                render_insight_panel(c, rank)

    # ------------------------------------------------------------------ SIDEBAR
    st.sidebar.title("🔎 CodeLens")
    st.sidebar.caption("Git-Aware Multi-Stage Code Retrieval Engine")

    repo_input = st.sidebar.text_input(
        "Repository Path or GitHub URL",
        value=S.get("repo_input", "data/demo_repo"),
        key="repo_input_widget",
        help="Enter local folder path or GitHub repository URL (e.g. https://github.com/psf/requests)",
    )

    c_btn1, c_btn2 = st.sidebar.columns([1, 1])
    with c_btn1:
        create_demo = st.button("⚡ Init Demo Repo", use_container_width=True)
    with c_btn2:
        max_commits = st.number_input("Max Commits", min_value=1, max_value=200, value=30)

    if create_demo:
        with st.spinner("Generating sample Git repository with history..."):
            demo_dir = Path("data/demo_repo").resolve()
            repo_input = create_demo_git_repo(str(demo_dir))
            S["repo_input"] = repo_input
            st.sidebar.success(f"Demo repo initialized at {Path(repo_input).relative_to(Path.cwd())}!")

    model_label = st.sidebar.selectbox("Embedding Model (CPU-Friendly)", list(MODELS))
    model_name = MODELS[model_label]

    if st.sidebar.button("🚀 Connect & Index Repository", type="primary", use_container_width=True):
        try:
            resolved_repo = ensure_repository_input(repo_input)
            repo_input = resolved_repo
            S["repo_input"] = resolved_repo
            with st.spinner("Fetching repository, extracting AST chunks, and indexing..."):
                t0 = time.perf_counter()
                local_path, current_sha, repo_obj = tracker.ingest_repository(resolved_repo, max_commits=max_commits)
                hist = build_history(local_path, max_commits=max_commits, repo_url=resolved_repo)

                if not hist.chunks:
                    fallback_requested = str(Path("data/demo_repo").resolve())
                    if Path(resolved_repo).exists() and Path(resolved_repo).name != "demo_repo":
                        st.sidebar.warning(
                            f"No readable code files were found in '{resolved_repo}'. Falling back to the demo repository."
                        )
                    resolved_repo = create_demo_git_repo(fallback_requested)
                    S["repo_input"] = resolved_repo
                    local_path, current_sha, repo_obj = tracker.ingest_repository(resolved_repo, max_commits=max_commits)
                    hist = build_history(local_path, max_commits=max_commits, repo_url=resolved_repo)
                    if not hist.chunks:
                        st.sidebar.error(
                            f"No supported code files were found in '{resolved_repo}' (.py, .js, .ts, .c, .cpp, .java)."
                        )
                    else:
                        pipe = Pipeline(hist.chunks, model_name=model_name)
                        links = link_versions(hist)
                        save_sqlite(hist, links)
                        elapsed = time.perf_counter() - t0
                        S.update(
                            repo_path=local_path,
                            repo_input=resolved_repo,
                            hist=hist,
                            pipe=pipe,
                            links=links,
                            index_secs=elapsed,
                            current_sha=current_sha,
                        )
                        st.sidebar.success(f"Indexed {len(hist.chunks)} chunks in {elapsed:.2f}s!")
                else:
                    pipe = Pipeline(hist.chunks, model_name=model_name)
                    links = link_versions(hist)
                    save_sqlite(hist, links)
                    elapsed = time.perf_counter() - t0
                    S.update(
                        repo_path=local_path,
                        repo_input=resolved_repo,
                        hist=hist,
                        pipe=pipe,
                        links=links,
                        index_secs=elapsed,
                        current_sha=current_sha,
                    )
                    st.sidebar.success(f"Indexed {len(hist.chunks)} chunks in {elapsed:.2f}s!")
        except Exception as e:
            st.sidebar.error(f"Indexing failed: {e}")

    if "hist" in S:
        st.sidebar.info(
            f"**Repository:** `{Path(S.repo_path).name}`\n\n"
            f"**Commit:** `{S.hist.latest_commit['sha'][:7]}`\n\n"
            f"**Chunks:** {len(S.hist.chunks)} · **Commits:** {len(S.hist.commits)}"
        )

    # ------------------------------------------------------------------ TABS
    tab_search, tab_monitor, tab_compare, tab_hist, tab_bench, tab_demo = st.tabs([
        "🔍 Multi-Stage Search",
        "📡 Index Monitor",
        "⚖️ Pipeline Compare",
        "🕰️ Code Evolution",
        "📊 Benchmarks",
        "🎬 Live Demo Workflow",
    ])

    ready = "pipe" in S

    # ------------------------------------------------------------------ TAB 1: Search
    with tab_search:
        st.subheader("Multi-Signal Code Retrieval")
        st.caption("Combines Query Intent Understanding, FAISS Semantic Vectors, BM25 Lexical, and Code Graph.")

        if not ready:
            st.info("👈 Connect and index a repository in the sidebar, or click **⚡ Init Demo Repo**.")
        else:
            q_col, opt_col = st.columns([3, 1])
            with q_col:
                sample_queries = [
                    "How is the input normalized before the main function?",
                    "Where is authentication handled?",
                    "Which function validates the user before login?",
                    "What code runs after authentication?",
                    "Which function calls verify_token?",
                    "What changed in the authentication code?",
                ]
                picked_query = st.selectbox("Quick query presets", ["Custom..."] + sample_queries)
                query_val = "" if picked_query == "Custom..." else picked_query

                query = st.text_input(
                    "Natural Language Query",
                    value=query_val,
                    placeholder="e.g. How is the input normalized before the main function?",
                )

            with opt_col:
                mode = st.selectbox("Retrieval Mode", list(MODE_LABELS), index=3, format_func=MODE_LABELS.get)
                version_mode = st.selectbox(
                    "Search Version Mode",
                    ["Current Version", "Specific Commit", "All History Versions"],
                )

            commit_filter_sha = None
            if version_mode == "Current Version":
                commit_filter_sha = S.hist.latest_commit["sha"]
            elif version_mode == "Specific Commit":
                n_commits = len(S.hist.commits)
                c_idx = st.selectbox(
                    "Select Target Commit",
                    range(n_commits - 1, -1, -1),
                    format_func=lambda i: commit_label(S.hist, i),
                )
                commit_filter_sha = S.hist.commits[c_idx]["sha"]
            else:
                commit_filter_sha = "ALL"

            if query:
                # Query Analysis & Intent Display
                q_analysis = analyze_query(query)
                col_intent1, col_intent2, col_intent3 = st.columns([2, 2, 2])
                with col_intent1:
                    st.markdown(
                        f"**Detected Intent:** `:{INTENT_COLORS.get(q_analysis.intent, 'gray')}[{q_analysis.intent}]`"
                    )
                with col_intent2:
                    syms_display = ", ".join([f"`{s}`" for s in q_analysis.symbols]) if q_analysis.symbols else "None"
                    st.markdown(f"**Extracted Symbols:** {syms_display}")
                with col_intent3:
                    rel_display = f"`{q_analysis.relationship_type}`" if q_analysis.relationship_type else "Direct"
                    st.markdown(f"**Structural Relationship:** {rel_display}")

                results, ms = run_search(query, mode=mode, k=10, commit_sha=commit_filter_sha)
                st.markdown(f"Found **{len(results)}** results in **{ms:.1f} ms**")

                if results:
                    show_results(results, "search")
                else:
                    st.warning("No code chunks matched the query under the selected version filter.")

    # ------------------------------------------------------------------ TAB 2: Index Monitor
    with tab_monitor:
        st.subheader("📡 Live Git Commit & Incremental Index Monitor")
        st.caption("Inspects real-time Git SHA changes, computes file diffs, and updates only affected chunks.")

        if not ready:
            st.info("Connect a repository first to monitor its Git index.")
        else:
            hist = S.hist
            latest_c = hist.latest_commit

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Commit SHA", latest_c["sha"][:7])
            m2.metric("Total Code Chunks", len(hist.chunks))
            m3.metric("Indexed Commits", len(hist.commits))
            m4.metric("Last Index Build", f"{S.get('index_secs', 0):.2f}s")

            st.markdown("---")
            st.markdown("#### 🔄 Trigger Git Commit Detection")
            col_sync1, col_sync2 = st.columns([2, 2])

            with col_sync1:
                if st.button("📥 Check & Pull Latest Commits (Incremental Update)", type="primary"):
                    with st.spinner("Checking repository for new commits..."):
                        hist, pipe, stats = tracker.incremental_index(S.hist, S.pipe, S.repo_path)
                        S.hist = hist
                        S.pipe = pipe
                        if stats.get("updated"):
                            st.success(f"New commit `{stats['sha'][:7]}` detected! Incrementally re-indexed.")
                            st.json(stats)
                        else:
                            st.info(stats.get("message", "No new commits found."))

            with col_sync2:
                st.markdown("**Webhook Receiver Status:**")
                st.code("POST http://localhost:8000/webhooks/github", language="text")
                st.caption("GitHub pushes automatically trigger incremental re-indexing via the REST API.")

            st.markdown("---")
            st.markdown("#### 📜 Indexed Commit History")
            commits_df = pd.DataFrame(hist.commits)
            st.dataframe(commits_df, hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------ TAB 3: Compare
    with tab_compare:
        st.subheader("⚖️ Ablation Comparison: Dense Baseline vs Full Pipeline")
        st.caption("Observe the ranking difference between pure embeddings and CodeLens multi-signal reranking.")

        if not ready:
            st.info("Index a repository first.")
        else:
            cmp_query = st.text_input(
                "Comparison Query",
                value="How is input normalized before the main function?",
                key="cmp_query_box",
            )
            if cmp_query:
                base_results, ms_base = run_search(cmp_query, "dense", 5)
                full_results, ms_full = run_search(cmp_query, "full", 5)

                base_rank_map = {r.chunk.id: idx for idx, r in enumerate(base_results, 1)}

                left_c, right_c = st.columns(2)
                with left_c:
                    st.markdown(f"### Mode A · Dense Embedding Only ({ms_base:.1f} ms)")
                    for idx, r in enumerate(base_results, 1):
                        st.markdown(
                            f"**{idx}.** `{r.chunk.name or 'code'}`  ·  `{r.chunk.path}:{r.chunk.start}`  "
                            f"(Score: {r.score:.3f})"
                        )

                with right_c:
                    st.markdown(f"### Mode D · CodeLens Full Pipeline ({ms_full:.1f} ms)")
                    for idx, r in enumerate(full_results, 1):
                        b_rank = base_rank_map.get(r.chunk.id)
                        if b_rank is None:
                            movement = "🆕 *Surfaced by Graph/Intent*"
                        elif b_rank > idx:
                            movement = f"▲ +{b_rank - idx} *Reranked Higher*"
                        elif b_rank < idx:
                            movement = f"▼ -{idx - b_rank}"
                        else:
                            movement = "= *Rank Match*"
                        st.markdown(
                            f"**{idx}.** `{r.chunk.name or 'code'}`  ·  `{r.chunk.path}:{r.chunk.start}`  "
                            f"(Score: {r.score:.3f})  —  {movement}"
                        )

    # ------------------------------------------------------------------ TAB 4: Evolution
    with tab_hist:
        st.subheader("🕰️ Evolutionary Code History")
        st.caption("Tracks function modifications, renames, additions, and deletions across commits.")

        if not ready:
            st.info("Index a repository with Git history to view code evolution.")
        elif len(S.hist.commits) < 2:
            st.info("Repository has only 1 commit. Multiple commits are required to view evolutionary links.")
        else:
            hist = S.hist
            links_data = []
            for i, l in enumerate(S.links):
                c_meta = hist.commits[l["commit_idx"]]
                old_chunk = hist.chunks[l["old"]] if l["old"] is not None else None
                new_chunk = hist.chunks[l["new"]] if l["new"] is not None else None
                path = (new_chunk or old_chunk).path if (new_chunk or old_chunk) else ""
                links_data.append({
                    "commit": c_meta["sha"][:7],
                    "date": c_meta["date"],
                    "change": l["kind"],
                    "old_symbol": old_chunk.name if old_chunk else "—",
                    "new_symbol": new_chunk.name if new_chunk else "—",
                    "file_path": path,
                    "token_similarity": l["sim"],
                    "_raw_idx": i,
                })

            df_links = pd.DataFrame(links_data)
            st.dataframe(df_links.drop(columns=["_raw_idx"]), hide_index=True, use_container_width=True)

            if len(df_links):
                st.markdown("#### 🔍 Inspect Change Diff")
                sel_change_idx = st.selectbox(
                    "Select a commit change to inspect",
                    range(len(df_links)),
                    format_func=lambda i: f"{df_links.loc[i, 'commit']}  {df_links.loc[i, 'change']}: "
                    f"{df_links.loc[i, 'old_symbol']} → {df_links.loc[i, 'new_symbol']} ({df_links.loc[i, 'file_path']})",
                )
                raw_link = S.links[df_links.loc[sel_change_idx, "_raw_idx"]]
                old_code = hist.chunks[raw_link["old"]].code if raw_link["old"] is not None else ""
                new_code = hist.chunks[raw_link["new"]].code if raw_link["new"] is not None else ""

                diff_output = "\n".join(
                    difflib.unified_diff(
                        old_code.splitlines(),
                        new_code.splitlines(),
                        "Before (Old)",
                        "After (New)",
                        lineterm="",
                    )
                )
                st.code(diff_output or "(No textual code change)", language="diff")

    # ------------------------------------------------------------------ TAB 5: Benchmark
    with tab_bench:
        st.subheader("📊 CoIR / AppsRetrieval Benchmark Evaluation")
        st.caption("Measures NDCG@10, MRR, Recall@10, and average query latency.")

        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            if st.button("▶️ Run Evaluation Benchmark Now", type="primary"):
                with st.spinner("Executing benchmark queries across 4 pipeline modes..."):
                    t_bench0 = time.perf_counter()
                    if "pipe" in S:
                        bench_df = evaluate_pipeline(S.pipe, DEFAULT_BENCHMARK_QUERIES)
                    else:
                        from codetrace.evaluate import run_evaluation
                        bench_df = run_evaluation(repo_path=".")
                    bench_df.to_csv("ablation_results.csv", index=False)
                    st.success(f"Benchmark finished in {time.perf_counter() - t_bench0:.1f}s!")

        results_file = Path("ablation_results.csv")
        if results_file.exists():
            df_bench = pd.read_csv(results_file)
            st.markdown("#### Evaluation Metrics Table")
            st.dataframe(df_bench, hide_index=True, use_container_width=True)

            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                st.markdown("##### Ranking Quality (NDCG@10 & MRR)")
                st.bar_chart(df_bench.set_index("mode")[["NDCG@10", "MRR"]])
            with chart_c2:
                st.markdown("##### Query Latency (ms)")
                st.bar_chart(df_bench.set_index("mode")["latency_ms"])
        else:
            st.info("Click **Run Evaluation Benchmark Now** or run `python evaluation/run_eval.py` in the terminal.")

    # ------------------------------------------------------------------ TAB 6: Demo Mode
    with tab_demo:
        st.subheader("🎬 Critical End-to-End Dynamic Demo")
        st.caption("Demonstrates the complete dynamic retrieval loop across code changes.")

        st.markdown(
            """
    **Demo Flow:**
    1. Query original repository for `"How is the input normalized before the main function?"`
    2. Inspect the retrieved function (`normalize_input`).
    3. Modify the code and commit directly to Git.
    4. CodeLens automatically detects the new commit.
    5. Only the changed file is incrementally re-indexed.
    6. Run the same query again -> The new implementation is immediately retrieved!
    7. Switch to Version History to observe both implementations.
    """
        )

        if not ready or not Path(S.repo_path).joinpath(".git").exists():
            st.warning("Please click **⚡ Init Demo Repo** in the sidebar first to load the test Git repository.")
        else:
            step_col1, step_col2 = st.columns(2)

            with step_col1:
                st.markdown("#### Step 1: Initial Query")
                demo_q = "How is the input normalized before the main function?"
                st.code(demo_q, language="text")

                if st.button("🔍 Run Initial Search", key="demo_btn1"):
                    init_res, ms_init = run_search(demo_q, mode="full", k=3)
                    if init_res:
                        top_chunk = init_res[0].chunk
                        st.success(f"Top Match: `{top_chunk.name}` at `{top_chunk.path}` (Commit: `{top_chunk.commit}`)")
                        st.code(top_chunk.code, language="python")

            with step_col2:
                st.markdown("#### Step 2: Make Code Change & Commit")
                st.markdown("Simulate a developer pushing an improved normalization function to Git:")

                new_code_snippet = '''def clean_data(raw_items):
    """Strip whitespace and remove empty strings from raw inputs."""
    return [x.strip() for x in raw_items if x]

def normalize_input(raw_input):
    """ADVANCED v2: Normalize input items with unicode stripping and regex sanitization."""
    import re
    cleaned = clean_data(raw_input)
    # New v2 logic: Strip non-alphanumeric and convert to lowercase
    return [re.sub(r'[^a-zA-Z0-9_]', '', item.lower()) for item in cleaned]
'''
                st.code(new_code_snippet, language="python")

                if st.button("💾 Commit Code Change & Trigger Incremental Index", type="primary", key="demo_btn2"):
                    try:
                        import git
                        repo = git.Repo(S.repo_path)
                        target_file = Path(S.repo_path) / "src" / "preprocess.py"
                        target_file.write_text(new_code_snippet, encoding="utf-8")
                        repo.index.add([str(target_file)])
                        author = git.Actor("Dev Lead", "dev@codelens.io")
                        new_commit = repo.index.commit("Upgrade normalize_input to v2 with regex sanitization", author=author, committer=author)

                        # Trigger incremental index
                        hist, pipe, stats = tracker.incremental_index(S.hist, S.pipe, S.repo_path, new_sha=new_commit.hexsha)
                        if stats.get("error"):
                            st.error(f"Incremental indexing failed: {stats['error']}")
                        else:
                            S.hist = hist
                            S.pipe = pipe
                            st.success(f"✨ New commit `{new_commit.hexsha[:7]}` detected and incrementally indexed in {stats['index_time_secs']:.3f}s!")
                            st.json(stats)
                    except Exception as ex:
                        st.error(f"Commit simulation failed: {ex}")

            st.markdown("---")
            st.markdown("#### Step 3: Verify Dynamic Retrieval After Commit")
            if st.button("🚀 Re-Run Search After Code Change", key="demo_btn3"):
                new_res, ms_new = run_search(demo_q, mode="full", k=3)
                if new_res:
                    top_new = new_res[0].chunk
                    st.success(f"Retrieved Updated Code: `{top_new.name}` (Commit `{top_new.commit}`):")
                    st.code(top_new.code, language="python")
                    st.info("Notice that the new v2 regex normalization is retrievable immediately, without full repository rebuilding!")


if __name__ == "__main__":
    main()
