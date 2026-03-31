"""Streamlit application — Pipeline B3: Detection d'Interactions Medicamenteuses Implicites.

Uses dspy.RLM with full REPL trajectory visualization, sub-LLM call tracking,
and code execution steps — same pattern as frontend.py and rlm_adk/frontend.py.
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

from rlm_medicaments.core.mcp_client import download_bdpm_files, clear_cache
from rlm_medicaments.core.bdpm_loader import (
    load_medications,
    load_compositions,
    filter_medications,
    generate_candidate_pairs,
    build_rcp_context,
)
from rlm_medicaments.core.rlm_pipeline import (
    InstrumentedRLM,
    PipelineConfig,
    build_rlm,
)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Interactions Medicamenteuses - RLM",
    page_icon="\U0001F48A",
    layout="centered",
)

MODELS = ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-5-mini"]

SUGGESTED_SEARCHES = [
    ("Cardiovasculaire", ["digoxine", "amiodarone", "aspirine", "clopidogrel", "amlodipine"]),
    ("Diabete", ["metformine", "glimepiride", "insuline", "sitagliptine"]),
    ("Psychotropes", ["sertraline", "alprazolam", "lithium", "olanzapine"]),
]

DEFAULT_QUERY = (
    "Analyse les interactions medicamenteuses implicites entre ces medicaments.\n\n"
    "METHODE OBLIGATOIRE — tu DOIS suivre ces etapes :\n\n"
    "1. Pour CHAQUE medicament, appelle llm_query() en lui passant le nom et la composition "
    "du medicament, et demande-lui d'agir en pharmacologue expert pour decrire :\n"
    "   - Le mecanisme d'action principal\n"
    "   - Les voies metaboliques (enzymes CYP450, transporteurs membranaires)\n"
    "   - Les cibles pharmacologiques\n"
    "   - Les organes affectes\n"
    "   - Les effets sur l'homeostasie (electrolytes, coagulation, pression arterielle)\n\n"
    "2. Pour CHAQUE paire de medicaments, appelle llm_query() en lui passant les deux "
    "profils mecanistiques, et demande-lui d'evaluer si une interaction implicite existe :\n"
    "   - Niveau de risque : aucun / faible / modere / eleve / critique\n"
    "   - Justification mecanistique detaillee\n"
    "   - Consequences cliniques potentielles\n\n"
    "3. Produis un RAPPORT FINAL en langage naturel structure avec :\n"
    "   - Un resume global\n"
    "   - Pour chaque interaction detectee (risque >= modere) : les deux medicaments, "
    "leurs profils mecanistiques, la justification de l'interaction, le niveau de risque\n"
    "   - Un avertissement que ces resultats sont des hypotheses a valider\n\n"
    "IMPORTANT : N'utilise PAS de simple comparaison de texte Python. "
    "Le raisonnement pharmacologique DOIT etre fait par llm_query(). "
    "Utilise llm_query_batched() quand tu as plusieurs appels independants."
)


@st.cache_resource
def _init_dspy():
    dspy.configure(lm=dspy.LM("openai/gpt-4o"))

_init_dspy()


# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background-color: #0e0e0e; color: #f0f0f0; }
  #MainMenu, footer, header { visibility: hidden; }

  [data-testid="stSidebar"] { background-color: #111 !important; border-right: 1px solid #1e1e1e; }
  [data-testid="stSidebar"] * { color: #ccc !important; }
  [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
  [data-testid="stSidebar"] .stNumberInput input,
  [data-testid="stSidebar"] .stTextInput input {
    background: #1a1a1a !important; border-color: #2a2a2a !important; color: #eee !important;
  }
  [data-testid="stSidebar"] label { color: #888 !important; font-size: 0.8rem !important; }
  [data-testid="stSidebar"] .sidebar-section {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #444 !important; margin: 1.2rem 0 0.5rem;
  }
  [data-testid="stSidebar"] hr { border-color: #1e1e1e !important; margin: 0.75rem 0; }

  .hero { text-align: center; padding: 2rem 0 1rem; }
  .hero h1 { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; color: #fff; margin-bottom: 0.3rem; }
  .hero p  { color: #888; font-size: 0.92rem; max-width: 700px; margin: 0 auto; }

  .stats { display: flex; justify-content: center; gap: 0.8rem; margin: 1rem 0 1.5rem; flex-wrap: wrap; }
  .stat  { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 999px;
            padding: 0.3rem 0.9rem; font-size: 0.8rem; color: #aaa; }
  .stat b { color: #fff; }

  .stTextArea textarea, .stTextInput > div > div > input {
    background: #1a1a1a !important; border: 1px solid #2e2e2e !important;
    border-radius: 10px !important; color: #f0f0f0 !important;
    font-size: 0.92rem !important; padding: 0.75rem !important;
  }
  .stTextArea textarea:focus, .stTextInput > div > div > input:focus {
    border-color: #555 !important; box-shadow: none !important;
  }

  .stButton > button {
    width: 100%; background: #ffffff !important; color: #000 !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    padding: 0.6rem !important; margin-top: 0.5rem; transition: opacity 0.15s;
  }
  .stButton > button:hover { opacity: 0.85; }
  .stButton > button:disabled { opacity: 0.3 !important; }

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

  .suggestion-chip {
    display: inline-block; background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 6px; padding: 0.3rem 0.7rem; margin: 0.2rem;
    font-size: 0.78rem; color: #aaa;
  }

  /* ── Trajectory ── */
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
        MODELS,
        index=0,
        help="LLM principal qui ecrit et execute le code dans le REPL.",
    )
    sub_model = st.selectbox(
        "Sous-modele",
        MODELS,
        index=1,
        help="LLM appele via llm_query() pour les sous-taches semantiques.",
    )

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Limites</div>', unsafe_allow_html=True)

    max_iterations = st.number_input(
        "Max iterations (steps REPL)",
        min_value=1, max_value=50, value=20,
        help="Nombre de tours code -> output dans le REPL.",
    )
    max_llm_calls = st.number_input(
        "Max appels LLM par run",
        min_value=1, max_value=200, value=50,
        help="Max llm_query() calls across all iterations.",
    )
    max_output_chars = st.number_input(
        "Troncature output (chars)",
        min_value=1000, max_value=500_000, value=100_000, step=10_000,
        help="Max chars de sortie REPL montres au LLM par step.",
    )

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Donnees</div>', unsafe_allow_html=True)

    commercialise_only = st.checkbox("Commercialises uniquement", value=True)

    if st.button("Vider le cache"):
        clear_cache()
        st.cache_data.clear()
        st.toast("Cache vide")


# ── Data loading ─────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_bdpm_data():
    files = download_bdpm_files()
    medications = []
    compositions = {}
    if "CIS_bdpm" in files:
        medications = load_medications(files["CIS_bdpm"])
    if "CIS_COMPO_bdpm" in files:
        compositions = load_compositions(files["CIS_COMPO_bdpm"])
    return medications, compositions


# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>\U0001F48A Interactions Medicamenteuses — RLM</h1>
  <p>Detection d'interactions implicites non documentees<br>
  Recursive Language Model &middot; DSPy &middot; BDPM data.gouv.fr</p>
</div>
""", unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────────────────────

with st.spinner("Chargement des donnees BDPM (MCP data.gouv.fr)..."):
    try:
        all_medications, compositions = load_bdpm_data()
    except Exception as e:
        st.error(f"Erreur chargement BDPM : {e}")
        st.stop()

if not all_medications:
    st.error("Aucun medicament charge.")
    st.stop()

st.markdown(f"""
<div class="stats">
  <span class="stat"><b>{len(all_medications):,}</b> medicaments BDPM</span>
  <span class="stat"><b>DSPy</b> RLM</span>
  <span class="stat"><b>{primary_model.split('/')[-1]}</b> &rarr; <b>{sub_model.split('/')[-1]}</b></span>
</div>
""", unsafe_allow_html=True)


# ── Session state for accumulated selections ────────────────────────────────

# Build lookup: CIS -> Medication object (stable across reruns)
_med_by_cis = {m.cis: m for m in all_medications}

if "selected_cis" not in st.session_state:
    st.session_state.selected_cis = []  # list of CIS codes


def _add_med(cis: str):
    if cis not in st.session_state.selected_cis:
        st.session_state.selected_cis.append(cis)


def _remove_med(cis: str):
    st.session_state.selected_cis = [c for c in st.session_state.selected_cis if c != cis]


def _clear_selection():
    st.session_state.selected_cis = []


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Recherche + Ajout
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="step-header"><span class="step-num">1</span> Rechercher et ajouter des medicaments</div>',
    unsafe_allow_html=True,
)

st.markdown("**Suggestions :** " + " &nbsp; ".join(
    f'<span class="suggestion-chip">{cat}: {", ".join(terms)}</span>'
    for cat, terms in SUGGESTED_SEARCHES
), unsafe_allow_html=True)

search_term = st.text_input(
    "Recherche par denomination",
    value="",
    placeholder="Tapez un nom de medicament (ex: digoxine, amiodarone, aspirine...)",
    label_visibility="collapsed",
)

if search_term:
    filtered_meds = filter_medications(all_medications, search_term=search_term, commercialise_only=commercialise_only)
    st.caption(f"{len(filtered_meds)} resultat(s) pour \"{search_term}\"")

    if filtered_meds:
        # Show results as selectable items (max 30 for perf)
        for med in filtered_meds[:30]:
            already = med.cis in st.session_state.selected_cis
            col_name, col_btn = st.columns([5, 1])
            with col_name:
                label = f"**{med.denomination}**"
                if already:
                    label += " &nbsp; ✓ selectionne"
                st.markdown(label, unsafe_allow_html=True)
            with col_btn:
                if already:
                    st.button("Retirer", key=f"rm_{med.cis}", on_click=_remove_med, args=(med.cis,))
                else:
                    st.button("Ajouter", key=f"add_{med.cis}", on_click=_add_med, args=(med.cis,))

        if len(filtered_meds) > 30:
            st.caption(f"... et {len(filtered_meds) - 30} autres. Affinez votre recherche.")
    else:
        st.warning(f"Aucun medicament pour \"{search_term}\".")
else:
    st.info("Tapez un nom de medicament ci-dessus pour le rechercher dans la base BDPM.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Selection actuelle
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="step-header"><span class="step-num">2</span> Medicaments selectionnes</div>',
    unsafe_allow_html=True,
)

# Resolve CIS codes to Medication objects
selected_meds = [_med_by_cis[cis] for cis in st.session_state.selected_cis if cis in _med_by_cis]

if selected_meds:
    # Show compact list of selected meds with remove buttons
    cols_per_row = 2
    for row_start in range(0, len(selected_meds), cols_per_row):
        row_meds = selected_meds[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, med in zip(cols, row_meds):
            with col:
                st.markdown(
                    f'<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;'
                    f'padding:0.5rem 0.8rem;margin-bottom:0.3rem;font-size:0.85rem;color:#ccc">'
                    f'{esc(med.denomination)}</div>',
                    unsafe_allow_html=True,
                )

    col_count, col_clear = st.columns([3, 1])
    with col_count:
        st.caption(f"{len(selected_meds)} medicament(s) selectionne(s)")
    with col_clear:
        st.button("Tout retirer", on_click=_clear_selection)
else:
    st.info(
        "Aucun medicament selectionne. Utilisez la recherche ci-dessus "
        "pour ajouter des medicaments de **differentes classes** (ex: digoxine + amiodarone + aspirine)."
    )
    st.stop()

if len(selected_meds) < 2:
    st.warning("Ajoutez au moins **2 medicaments** pour lancer l'analyse.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Question
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="step-header"><span class="step-num">3</span> Question d\'analyse</div>',
    unsafe_allow_html=True,
)

user_query = st.text_area(
    "Query",
    value=DEFAULT_QUERY,
    height=100,
    label_visibility="collapsed",
    help="Cette question guide le raisonnement du RLM.",
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Lancement
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="step-header"><span class="step-num">4</span> Lancer le RLM</div>',
    unsafe_allow_html=True,
)

# Build context: all selected medications with their composition data
context_parts = []
for med in selected_meds:
    rcp = build_rcp_context(med, compositions)
    context_parts.append(rcp)

context_text = (
    "=== DONNEES BDPM (Base de Donnees Publique des Medicaments — data.gouv.fr) ===\n\n"
    f"Nombre de medicaments a analyser : {len(selected_meds)}\n\n"
    "ATTENTION : Les donnees ci-dessous ne contiennent que la COMPOSITION (substances actives).\n"
    "Elles ne contiennent PAS les mecanismes d'action, les voies metaboliques, ni la pharmacodynamie.\n"
    "Tu DOIS utiliser llm_query() pour obtenir le profil mecanistique de chaque medicament\n"
    "en te basant sur tes connaissances pharmacologiques.\n\n"
    + "\n\n---\n\n".join(context_parts)
)

# Build pairs info for display
pairs = generate_candidate_pairs(selected_meds, compositions)
n_pairs = len(pairs)

st.markdown(f"""
<div class="stats">
  <span class="stat"><b>{len(selected_meds)}</b> medicaments</span>
  <span class="stat"><b>{n_pairs}</b> paires candidates</span>
  <span class="stat"><b>{len(context_text):,}</b> chars de contexte</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
  <strong>Avertissement :</strong> Les resultats sont des <em>hypotheses a valider
  par des professionnels de sante</em>. Ils ne constituent pas un avis medical.
</div>
""", unsafe_allow_html=True)

run_btn = st.button("Run", disabled=not user_query.strip())


# ═══════════════════════════════════════════════════════════════════════════════
# Execution RLM + Trajectory
# ═══════════════════════════════════════════════════════════════════════════════

if run_btn:
    config = PipelineConfig(
        primary_model=primary_model,
        sub_model=sub_model,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        max_output_chars=max_output_chars,
        user_query=user_query,
    )
    rlm = build_rlm(config)

    with st.spinner("RLM en cours..."):
        with dspy.context(lm=dspy.LM(primary_model)):
            result = rlm(context=context_text, query=user_query)

    trajectory = result.trajectory or []
    n = len(trajectory)

    # ── Trajectory REPL ──────────────────────────────────────────────────────

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
        sub_calls_count = code.count("llm_query(") + code.count("llm_query_batched(")
        if sub_calls_count > 1:
            badge += f' <span class="badge badge-llm">{sub_calls_count}x calls</span>'

        # ── Dot + Body columns ───────────────────────────────────────────────
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

            # Sub-LLM calls captured by InstrumentedRLM
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

    total_subcalls = sum(len(v) for v in rlm.subcalls.values())

    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Resultat final &mdash; {n} iterations, {total_subcalls} sous-appels LLM</div>
      {result.answer.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────

    export_data = {
        "query": user_query,
        "answer": result.answer,
        "config": {
            "primary_model": config.primary_model,
            "sub_model": config.sub_model,
            "max_iterations": config.max_iterations,
            "max_llm_calls": config.max_llm_calls,
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
        "Telecharger le rapport (JSON)",
        data=json.dumps(export_data, ensure_ascii=False, indent=2),
        file_name="b3_rlm_report.json",
        mime="application/json",
    )
