import html
import os
import sys
from datetime import datetime

# Ensure parent directory is on sys.path so rlm_adk is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from rlm_adk.agent import RLMAgent

MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "lex_fridman_dataset.csv")
DEFAULT_QUERY = (
    "Find what the first 5 Machine Learning guests had to say about AGI "
    "in the Lex Fridman Podcast. Not all guests are ML guests — focus on "
    "established researchers known for their contribution to AI/ML. "
    "Return a summary per guest about what they said about AGI."
)

st.set_page_config(page_title="Lex Fridman RLM Demo (ADK)", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
  .stApp { background-color: #0e0e0e; color: #f0f0f0; }
  #MainMenu, footer, header { visibility: hidden; }

  /* Sidebar */
  [data-testid="stSidebar"] { background-color: #111 !important; border-right: 1px solid #1e1e1e; }
  [data-testid="stSidebar"] * { color: #ccc !important; }
  [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
  [data-testid="stSidebar"] .stNumberInput input {
    background: #1a1a1a !important; border-color: #2a2a2a !important; color: #eee !important;
  }
  [data-testid="stSidebar"] label { color: #888 !important; font-size: 0.8rem !important; }
  [data-testid="stSidebar"] .sidebar-section {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #444 !important; margin: 1.2rem 0 0.5rem;
  }
  [data-testid="stSidebar"] hr { border-color: #1e1e1e !important; margin: 0.75rem 0; }

  .hero { text-align: center; padding: 2.5rem 0 1.5rem; }
  .hero h1 { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px; color: #fff; margin-bottom: 0.3rem; }
  .hero p  { color: #888; font-size: 0.95rem; }

  .stats { display: flex; justify-content: center; gap: 1rem; margin: 1.2rem 0 2rem; flex-wrap: wrap; }
  .stat  { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 999px;
            padding: 0.35rem 1rem; font-size: 0.82rem; color: #aaa; }
  .stat b { color: #fff; }

  .stTextArea textarea {
    background: #1a1a1a !important; border: 1px solid #2e2e2e !important;
    border-radius: 10px !important; color: #f0f0f0 !important;
    font-size: 0.92rem !important; padding: 0.75rem !important;
  }
  .stTextArea textarea:focus { border-color: #555 !important; box-shadow: none !important; }

  .stButton > button {
    width: 100%; background: #ffffff !important; color: #000 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    padding: 0.6rem !important; margin-top: 0.5rem; transition: opacity 0.15s;
  }
  .stButton > button:hover { opacity: 0.85; }
  .stButton > button:disabled { opacity: 0.3 !important; }

  /* Trajectory */
  .traj-header {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #444; margin: 2.5rem 0 1.2rem;
  }
  .step-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #555; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .badge { font-size: 0.65rem; padding: 0.15rem 0.55rem; border-radius: 999px; font-weight: 700; letter-spacing: 0.05em; }
  .badge-llm   { background: #2e1065; color: #a78bfa; border: 1px solid #4c1d95; }
  .badge-final { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
  .badge-adk   { background: #1a1a3e; color: #818cf8; border: 1px solid #312e81; }

  .reasoning-box {
    background: #111; border: 1px solid #1e1e1e; border-radius: 8px;
    padding: 0.7rem 0.9rem; font-size: 0.83rem; color: #666;
    line-height: 1.6; margin-bottom: 0.6rem; font-style: italic;
  }
  .output-box {
    background: #0a0a0a; border: 1px solid #1e1e1e; border-radius: 8px;
    padding: 0.7rem 0.9rem; font-size: 0.80rem; color: #777;
    font-family: monospace; white-space: pre-wrap; word-break: break-word;
    max-height: 200px; overflow-y: auto; margin-top: 0.5rem;
  }

  /* Sub-call cards */
  .subcall-card {
    background: #150d2a; border: 1px solid #3b1f6e;
    border-left: 3px solid #7c3aed; border-radius: 6px;
    padding: 0.6rem 0.85rem; margin-bottom: 0.4rem;
  }
  .subcall-header { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; color: #7c3aed; text-transform: uppercase; margin-bottom: 0.4rem; }
  .subcall-row { display: flex; gap: 0.5rem; font-size: 0.78rem; line-height: 1.5; }
  .subcall-lbl { color: #555; flex-shrink: 0; width: 4rem; }
  .subcall-txt { color: #999; word-break: break-word; }

  /* Result card */
  .result-card {
    background: #0d1f12; border: 1px solid #1a3a22;
    border-radius: 12px; padding: 1.5rem 1.75rem;
    margin: 1.5rem 0; line-height: 1.75; font-size: 0.93rem; color: #ccc;
  }
  .result-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; color: #2d6a3f; text-transform: uppercase; margin-bottom: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar config ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")

    st.markdown('<div class="sidebar-section">Models</div>', unsafe_allow_html=True)
    primary_model = st.selectbox("Primary agent", MODELS, index=0,
                                 help="Main Gemini model that writes and executes code in the REPL.")
    sub_model     = st.selectbox("Sub-agent", MODELS, index=0,
                                 help="Gemini model called via llm_query() for semantic sub-tasks.")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Limits</div>', unsafe_allow_html=True)
    max_iterations = st.number_input("Max iterations (REPL steps)", min_value=1, max_value=50, value=20,
                                     help="Maximum number of code→output REPL iterations.")
    max_llm_calls  = st.number_input("Max LLM calls per run", min_value=1, max_value=200, value=50,
                                     help="Max llm_query() calls across all iterations.")
    max_output_chars = st.number_input("Truncate length (chars)", min_value=1000, max_value=500_000,
                                       value=100_000, step=10_000,
                                       help="Max REPL output chars shown to the LLM per step.")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">API</div>', unsafe_allow_html=True)
    api_key = st.text_input("Google API Key", value=os.environ.get("GOOGLE_API_KEY", ""),
                            type="password", help="Gemini API key (or set GOOGLE_API_KEY env var)")


@st.cache_resource(show_spinner="Loading 319 transcripts…")
def load_data():
    with open(DATA_FILE, encoding="utf-8", errors="replace") as f:
        csv_text = f.read()
    return (
        "The following is a CSV file with transcripts from the Lex Fridman Podcast.\n"
        "Columns: id, guest, title, text\n\nCSV data starts below:\n" + csv_text
    )


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🎙️ RLM Demo — ADK</h1>
  <p>Recursive Language Model querying 319 podcast transcripts</p>
</div>
<div class="stats">
  <span class="stat"><b>319</b> episodes</span>
  <span class="stat"><b>37M</b> chars</span>
  <span class="stat"><b>Google ADK</b> + Gemini</span>
  <span class="stat"><b>{primary_model}</b> → <b>{sub_model}</b></span>
</div>
""", unsafe_allow_html=True)

query = st.text_area("Query", placeholder="Ask anything about the podcast…",
                     value=DEFAULT_QUERY, height=130, label_visibility="collapsed")

run = st.button("Run", disabled=not query.strip() or not api_key)

if not api_key:
    st.caption("Enter your Google API key in the sidebar or set GOOGLE_API_KEY in .env")

# ── Run ────────────────────────────────────────────────────────────────────────
if run:
    try:
        context = load_data()
    except FileNotFoundError:
        st.error(f"`{DATA_FILE}` not found — run `python download_data.py` first.")
        st.stop()

    agent = RLMAgent(
        name="rlm_agent",
        model=primary_model,
        sub_model=sub_model,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        max_output_chars=max_output_chars,
        api_key=api_key,
    )

    with st.spinner("RLM running with Google ADK + Gemini…"):
        result = agent.run(context=context, query=query)

    # ── Auto-export ────────────────────────────────────────────────────────────
    runs_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_path = result.export_markdown(os.path.join(runs_dir, f"run_{ts}.md"))
    st.toast(f"Run saved to {export_path.name}", icon="💾")

    trajectory = result.trajectory
    n = len(trajectory)

    # ── Trajectory ─────────────────────────────────────────────────────────────
    st.markdown('<div class="traj-header">REPL Trajectory</div>', unsafe_allow_html=True)

    def esc(s: str) -> str:
        return html.escape(s, quote=False)

    for i, step in enumerate(trajectory):
        is_last   = (i == n - 1)
        has_llm   = bool(step.subcalls)

        dot_color = "#16a34a" if is_last else ("#7c3aed" if has_llm else "#444")
        badge = '<span class="badge badge-adk">ADK</span>'
        if has_llm:
            badge += ' <span class="badge badge-llm">sub-LLM call</span>'
        if is_last:
            badge += ' <span class="badge badge-final">SUBMIT</span>'
        if len(step.subcalls) > 1:
            badge += f' <span class="badge badge-llm">{len(step.subcalls)}× calls</span>'

        col_dot, col_body = st.columns([0.03, 0.97])

        with col_dot:
            connector_style = "" if is_last else (
                "border-left: 2px solid #222; margin-left: 4px; min-height: 300px;"
            )
            st.markdown(
                f'<div style="display:flex;flex-direction:column;align-items:center;padding-top:4px">'
                f'  <div style="width:10px;height:10px;border-radius:50%;background:{dot_color};flex-shrink:0"></div>'
                f'  <div style="{connector_style}"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_body:
            st.markdown(
                f'<div class="step-label">Step {step.step} / {n} &nbsp;{badge}</div>'
                f'<div class="reasoning-box">{esc(step.reasoning)}</div>',
                unsafe_allow_html=True,
            )
            if step.code:
                st.code(step.code, language="python")

            for j, sc in enumerate(step.subcalls):
                st.markdown(
                    f'<div class="subcall-card">'
                    f'  <div class="subcall-header">Sub-LLM call #{j+1}</div>'
                    f'  <div class="subcall-row"><span class="subcall-lbl">Prompt</span>'
                    f'    <span class="subcall-txt">{esc(sc.prompt[:300])}…</span></div>'
                    f'  <div class="subcall-row"><span class="subcall-lbl">Response</span>'
                    f'    <span class="subcall-txt">{esc(sc.response[:300])}…</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if step.output:
                st.markdown(
                    f'<div class="output-box">{esc(step.output)}</div>',
                    unsafe_allow_html=True,
                )

    # ── Final answer ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Final Result — {result.total_iterations} iterations, {result.total_subcalls} sub-LLM calls</div>
      {result.answer.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
