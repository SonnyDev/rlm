"""Streamlit — Suivi des Recommandations de la Cour des Comptes par RLM.

Utilise dspy.RLM avec trajectoire REPL complete, sous-appels LLM traces,
et visualisation identique a frontend.py / rlm_adk/frontend.py.
"""

import html
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from dotenv import load_dotenv
import dspy

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from rlm_cours_de_comptes.core.mcp_client import (
    download_all_reports,
    download_recommendations,
    clear_cache,
)
from rlm_cours_de_comptes.core.corpus_loader import (
    THEMES,
    filter_reports_by_theme,
    filter_reports_by_period,
    build_rlm_context,
    summarize_corpus,
)
from rlm_cours_de_comptes.core.rlm_pipeline import (
    PipelineConfig,
    DEFAULT_QUERY,
    SUGGESTED_QUERIES,
    build_rlm,
)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cour des Comptes - RLM",
    page_icon="\U0001F3DB",
    layout="centered",
)

MODELS = [
    "gemini/gemini-3.1-flash-lite-preview",
    "gemini/gemini-2.5-flash",
    "openai/gpt-5-mini",
    "openai/gpt-5",
]


@st.cache_resource
def _init_dspy():
    dspy.configure(lm=dspy.LM("gemini/gemini-3.1-flash-lite-preview"))

_init_dspy()


# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background-color: #0e0e0e; color: #f0f0f0; }
  #MainMenu, footer, header { visibility: hidden; }

  [data-testid="stSidebar"] { background-color: #111 !important; border-right: 1px solid #1e1e1e; }
  [data-testid="stSidebar"] * { color: #ccc !important; }
  [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
  [data-testid="stSidebar"] .stNumberInput input {
    background: #1a1a1a !important; border-color: #2a2a2a !important; color: #eee !important;
  }
  [data-testid="stSidebar"] label { color: #888 !important; font-size: 0.8rem !important; }
  [data-testid="stSidebar"] .stButton > button {
    background: #1a1a1a !important; color: #ccc !important;
    border: 1px solid #333 !important; height: auto !important; min-height: auto !important;
  }
  [data-testid="stSidebar"] .stButton > button:hover { background: #2a2a2a !important; }
  [data-testid="stSidebar"] .sidebar-section {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #444 !important; margin: 1.2rem 0 0.5rem;
  }
  [data-testid="stSidebar"] hr { border-color: #1e1e1e !important; margin: 0.75rem 0; }

  .block-container { padding-top: 1rem !important; }
  .hero { text-align: center; padding: 0 0 1rem; }
  .hero h1 { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; color: #fff; margin-bottom: 0.3rem; }
  .hero p  { color: #888; font-size: 0.92rem; max-width: 700px; margin: 0 auto; }

  .stats { display: flex; justify-content: center; gap: 0.8rem; margin: 1rem 0 1.5rem; flex-wrap: wrap; }
  .stat  { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 999px;
            padding: 0.3rem 0.9rem; font-size: 0.8rem; color: #aaa; }
  .stat b { color: #fff; }

  .stTextArea textarea {
    background: #1a1a1a !important; border: 1px solid #2e2e2e !important;
    border-radius: 10px !important; color: #f0f0f0 !important;
    font-size: 0.92rem !important; padding: 0.75rem !important;
  }
  .stTextArea textarea:focus { border-color: #555 !important; box-shadow: none !important; }

  .stButton > button {
    width: 100% !important; box-sizing: border-box !important;
    background: #ffffff !important; color: #000 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 0.6rem 0.8rem !important; margin-top: 0.5rem; transition: opacity 0.15s;
    min-height: 3.6rem !important; height: 3.6rem !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    text-align: center !important; white-space: normal !important; line-height: 1.3 !important;
  }
  .stButton > button:hover { opacity: 0.85; }
  .stButton > button:disabled { opacity: 0.3 !important; }
  .stDownloadButton > button {
    background: #1a1a1a !important; color: #fff !important;
    border: 1px solid #333 !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 0.6rem 0.8rem !important;
  }
  .stDownloadButton > button:hover { background: #2a2a2a !important; }

  .step-header {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #555; margin: 2rem 0 0.8rem;
    display: flex; align-items: center; gap: 0.6rem;
  }
  .step-num {
    background: #1a1a1a; border: 1px solid #333; border-radius: 50%;
    width: 24px; height: 24px; display: inline-flex; align-items: center;
    justify-content: center; font-size: 0.7rem; color: #aaa; flex-shrink: 0;
  }

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
  .badge { font-size: 0.65rem; padding: 0.15rem 0.55rem; border-radius: 999px;
           font-weight: 700; letter-spacing: 0.05em; }
  .badge-llm   { background: #2e1065; color: #a78bfa; border: 1px solid #4c1d95; }
  .badge-final { background: #052e16; color: #4ade80; border: 1px solid #14532d; }

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

  .subcall-card {
    background: #150d2a; border: 1px solid #3b1f6e;
    border-left: 3px solid #7c3aed; border-radius: 6px;
    padding: 0.6rem 0.85rem; margin-bottom: 0.4rem;
  }
  .subcall-header { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
                    color: #7c3aed; text-transform: uppercase; margin-bottom: 0.4rem; }
  .subcall-row { display: flex; gap: 0.5rem; font-size: 0.78rem; line-height: 1.5; }
  .subcall-lbl { color: #555; flex-shrink: 0; width: 4rem; }
  .subcall-txt { color: #999; word-break: break-word; }

  .result-card {
    background: #0d1f12; border: 1px solid #1a3a22;
    border-radius: 12px; padding: 1.5rem 1.75rem;
    margin: 1.5rem 0; line-height: 1.75; font-size: 0.93rem; color: #ccc;
  }
  .result-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
                  color: #2d6a3f; text-transform: uppercase; margin-bottom: 0.75rem; }

  .warning-box {
    background: #1a1a0a; border: 1px solid #3a3a1a; border-left: 3px solid #d97706;
    border-radius: 8px; padding: 0.8rem 1rem; margin: 1rem 0;
    font-size: 0.82rem; color: #cca; line-height: 1.5;
  }
</style>
""", unsafe_allow_html=True)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Configuration")

    st.markdown('<div class="sidebar-section">Modeles</div>', unsafe_allow_html=True)
    primary_model = st.selectbox(
        "Modele principal",
        MODELS, index=0,
        help="LLM principal (ecrit le code REPL + appel racine d'agregation).",
    )
    sub_model = st.selectbox(
        "Sous-modele",
        MODELS, index=1,
        help="LLM pour les sous-appels (extraction par rapport).",
    )

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Limites RLM</div>', unsafe_allow_html=True)

    max_iterations = st.number_input("Max iterations REPL", min_value=1, max_value=50, value=25)
    max_llm_calls = st.number_input("Max appels LLM", min_value=1, max_value=200, value=80)
    max_output_chars = st.number_input("Troncature output", min_value=1000, max_value=500_000, value=100_000, step=10_000)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Cache</div>', unsafe_allow_html=True)

    if st.button("Vider le cache"):
        clear_cache()
        st.cache_data.clear()
        st.toast("Cache vide")


# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>\U0001F3DB Cour des Comptes — RLM</h1>
  <p>Suivi des recommandations recurrentes par Recursive Language Model<br>
  Corpus de rapports publics &middot; DSPy &middot; data.gouv.fr</p>
</div>
""", unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_all_data():
    reports = download_all_reports()
    recommendations = download_recommendations()
    return reports, recommendations


with st.spinner("Chargement des rapports depuis data.gouv.fr (MCP)..."):
    try:
        all_reports, ground_truth_recs = load_all_data()
    except Exception as e:
        st.error(f"Erreur chargement : {e}")
        st.stop()

if not all_reports:
    st.error("Aucun rapport charge.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Theme & Period
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="step-header"><span class="step-num">1</span> Theme et periode</div>',
    unsafe_allow_html=True,
)

col_theme, col_period = st.columns([1, 1])

with col_theme:
    theme = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )

with col_period:
    available_years = sorted(set(r["year"] for r in all_reports))
    # Default to 2015-2016: intersection between reports (2013-2016) and recommendations (2015-2018)
    default_min = 2015 if 2015 in available_years else min(available_years)
    default_max = 2016 if 2016 in available_years else max(available_years)
    year_min, year_max = st.select_slider(
        "Periode",
        options=available_years,
        value=(default_min, default_max),
    )

# Apply filters
filtered = filter_reports_by_theme(all_reports, theme)
filtered = filter_reports_by_period(filtered, year_min, year_max)

if not filtered:
    st.warning(f"Aucun rapport trouve pour le theme \"{theme}\" sur {year_min}-{year_max}.")
    st.stop()

stats = summarize_corpus(filtered)

st.markdown(f"""
<div class="stats">
  <span class="stat"><b>{stats['total_reports']}</b> rapports</span>
  <span class="stat"><b>{stats['total_chars']:,}</b> caracteres</span>
  <span class="stat">~<b>{stats['total_tokens_approx']:,}</b> tokens</span>
  <span class="stat"><b>{primary_model.split('/')[-1]}</b> &rarr; <b>{sub_model.split('/')[-1]}</b></span>
</div>
""", unsafe_allow_html=True)

# Corpus details
with st.expander(f"Corpus : {stats['total_reports']} rapports"):
    for year, count in sorted(stats["by_year"].items()):
        st.markdown(f"**{year}** : {count} rapport(s)")
    st.caption(f"Volume total : {stats['total_chars']:,} caracteres (~{stats['total_tokens_approx']:,} tokens)")

# Ground truth info
gt_count = len(ground_truth_recs)
st.markdown(f"""
<div class="stats">
  <span class="stat"><b>{gt_count}</b> recommandations officielles (2015-2018, verite terrain)</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Query
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="step-header"><span class="step-num">2</span> Question d\'analyse</div>',
    unsafe_allow_html=True,
)

# Query suggestions as clickable buttons
if "user_query" not in st.session_state:
    st.session_state.user_query = DEFAULT_QUERY

st.markdown("**Questions suggerées :**")
row1 = st.columns(3)
for col, (label, query_text) in zip(row1, SUGGESTED_QUERIES[:3]):
    with col:
        if st.button(label, key=f"sq_{label}"):
            st.session_state.user_query = query_text
row2 = st.columns(3)
for col, (label, query_text) in zip(row2, SUGGESTED_QUERIES[3:6]):
    with col:
        if st.button(label, key=f"sq_{label}"):
            st.session_state.user_query = query_text

user_query = st.text_area(
    "Query",
    value=st.session_state.user_query,
    height=100,
    label_visibility="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Run
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="step-header"><span class="step-num">3</span> Lancer le RLM</div>',
    unsafe_allow_html=True,
)

run_btn = st.button("Run", disabled=not user_query.strip())

if run_btn:
    # Build context
    context_text = build_rlm_context(filtered)

    config = PipelineConfig(
        primary_model=primary_model,
        sub_model=sub_model,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        max_output_chars=max_output_chars,
    )
    rlm = build_rlm(config)

    with st.spinner(f"RLM en cours sur {stats['total_reports']} rapports..."):
        with dspy.context(lm=dspy.LM(primary_model)):
            result = rlm(context=context_text, query=user_query)

    trajectory = result.trajectory or []
    n = len(trajectory)
    total_subcalls = sum(len(v) for v in rlm.subcalls.values())

    # ── Trajectory ───────────────────────────────────────────────────────────

    st.markdown('<div class="traj-header">Trajectoire REPL</div>', unsafe_allow_html=True)

    for i, step in enumerate(trajectory):
        code      = step.get("code", "")
        output    = step.get("output", "")
        reasoning = step.get("reasoning", "")
        is_last   = (i == n - 1)
        has_llm   = "llm_query" in code

        dot_color = "#16a34a" if is_last else ("#7c3aed" if has_llm else "#444")
        badge = ""
        if has_llm:
            badge = '<span class="badge badge-llm">sub-LLM call</span>'
        if is_last:
            badge = '<span class="badge badge-final">SUBMIT</span>'
        sub_count = code.count("llm_query(") + code.count("llm_query_batched(")
        if sub_count > 1:
            badge += f' <span class="badge badge-llm">{sub_count}x calls</span>'

        col_dot, col_body = st.columns([0.03, 0.97])

        with col_dot:
            connector = "" if is_last else "border-left: 2px solid #222; margin-left: 4px; min-height: 300px;"
            st.markdown(
                f'<div style="display:flex;flex-direction:column;align-items:center;padding-top:4px">'
                f'  <div style="width:10px;height:10px;border-radius:50%;background:{dot_color};flex-shrink:0"></div>'
                f'  <div style="{connector}"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_body:
            st.markdown(
                f'<div class="step-label">Step {i+1} / {n} &nbsp;{badge}</div>'
                f'<div class="reasoning-box">{esc(reasoning)}</div>',
                unsafe_allow_html=True,
            )

            if code:
                st.code(code, language="python")

            subcalls = rlm.subcalls.get(i, [])
            for j, sc in enumerate(subcalls):
                prompt_preview = esc(sc["prompt"][:500])
                response_preview = esc(sc["response"][:500])
                st.markdown(
                    f'<div class="subcall-card">'
                    f'  <div class="subcall-header">Sub-LLM call #{j+1}</div>'
                    f'  <div class="subcall-row"><span class="subcall-lbl">Prompt</span>'
                    f'    <span class="subcall-txt">{prompt_preview}{"..." if len(sc["prompt"]) > 500 else ""}</span></div>'
                    f'  <div class="subcall-row"><span class="subcall-lbl">Response</span>'
                    f'    <span class="subcall-txt">{response_preview}{"..." if len(sc["response"]) > 500 else ""}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if output:
                st.markdown(
                    f'<div class="output-box">{esc(output)}</div>',
                    unsafe_allow_html=True,
                )

    # ── Final answer ─────────────────────────────────────────────────────────

    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Resultat final &mdash; {n} iterations, {total_subcalls} sous-appels LLM</div>
      {result.answer.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────

    st.markdown("---")

    export_data = {
        "theme": theme,
        "period": f"{year_min}-{year_max}",
        "corpus": {
            "total_reports": stats["total_reports"],
            "total_chars": stats["total_chars"],
        },
        "query": user_query,
        "answer": result.answer,
        "config": {
            "primary_model": primary_model,
            "sub_model": sub_model,
            "max_iterations": max_iterations,
            "max_llm_calls": max_llm_calls,
        },
        "trajectory": [
            {
                "step": i + 1,
                "reasoning": step.get("reasoning", ""),
                "code": step.get("code", ""),
                "output": step.get("output", ""),
                "subcalls": [
                    {"prompt": sc["prompt"], "response": sc["response"]}
                    for sc in rlm.subcalls.get(i, [])
                ],
            }
            for i, step in enumerate(trajectory)
        ],
    }

    st.download_button(
        "Telecharger le rapport complet (JSON)",
        data=json.dumps(export_data, ensure_ascii=False, indent=2),
        file_name="cour_des_comptes_rlm_report.json",
        mime="application/json",
    )
