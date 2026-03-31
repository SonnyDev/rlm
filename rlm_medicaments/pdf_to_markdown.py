"""Streamlit app — Convert multiple PDFs to a single Markdown file using MarkItDown."""

import io
import tempfile
import os

import streamlit as st
from markitdown import MarkItDown

st.set_page_config(
    page_title="PDF to Markdown — MarkItDown",
    page_icon="\U0001F4C4",
    layout="centered",
)

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background-color: #0e0e0e; color: #f0f0f0; }
  #MainMenu, footer, header { visibility: hidden; }

  .hero { text-align: center; padding: 2rem 0 1rem; }
  .hero h1 { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; color: #fff; margin-bottom: 0.3rem; }
  .hero p  { color: #888; font-size: 0.92rem; }

  .stats { display: flex; justify-content: center; gap: 0.8rem; margin: 1rem 0 1.5rem; flex-wrap: wrap; }
  .stat  { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 999px;
            padding: 0.3rem 0.9rem; font-size: 0.8rem; color: #aaa; }
  .stat b { color: #fff; }

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

  .file-card {
    background: #111; border: 1px solid #1e1e1e; border-radius: 8px;
    padding: 0.5rem 0.8rem; margin-bottom: 0.3rem;
    font-size: 0.85rem; color: #ccc;
    display: flex; justify-content: space-between; align-items: center;
  }
  .file-size { color: #666; font-size: 0.75rem; }

  .result-card {
    background: #0d1f12; border: 1px solid #1a3a22;
    border-radius: 12px; padding: 1.5rem 1.75rem;
    margin: 1.5rem 0; line-height: 1.75; font-size: 0.93rem; color: #ccc;
  }
  .result-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
                  color: #2d6a3f; text-transform: uppercase; margin-bottom: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>\U0001F4C4 PDF to Markdown</h1>
  <p>Convertissez plusieurs PDFs en un seul fichier Markdown avec MarkItDown</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="step-header"><span class="step-num">1</span> Uploader des fichiers PDF</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    total_size = sum(f.size for f in uploaded_files)
    st.markdown(f"""
    <div class="stats">
      <span class="stat"><b>{len(uploaded_files)}</b> fichier{"s" if len(uploaded_files) > 1 else ""}</span>
      <span class="stat"><b>{total_size / 1024 / 1024:.1f}</b> Mo au total</span>
      <span class="stat"><b>MarkItDown</b></span>
    </div>
    """, unsafe_allow_html=True)

    for f in uploaded_files:
        size_kb = f.size / 1024
        unit = "Ko" if size_kb < 1024 else "Mo"
        size_display = f"{size_kb:.0f} {unit}" if size_kb < 1024 else f"{size_kb/1024:.1f} {unit}"
        st.markdown(
            f'<div class="file-card"><span>{f.name}</span><span class="file-size">{size_display}</span></div>',
            unsafe_allow_html=True,
        )
else:
    st.info("Glissez-deposez vos fichiers PDF ci-dessus ou cliquez pour parcourir.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Convert
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="step-header"><span class="step-num">2</span> Convertir</div>',
    unsafe_allow_html=True,
)

convert_btn = st.button(f"Convertir {len(uploaded_files)} PDF{'s' if len(uploaded_files) > 1 else ''} en Markdown")

if convert_btn:
    md = MarkItDown()
    markdown_parts: list[str] = []
    errors: list[str] = []

    progress = st.progress(0, text="Conversion en cours...")

    for i, uploaded in enumerate(uploaded_files):
        progress.progress((i) / len(uploaded_files), text=f"Conversion de {uploaded.name}...")

        try:
            # MarkItDown needs a file path — write to a temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            result = md.convert(tmp_path)
            text = result.text_content.strip()

            if text:
                markdown_parts.append(f"# {uploaded.name}\n\n{text}")
            else:
                markdown_parts.append(f"# {uploaded.name}\n\n*Aucun contenu extrait.*")

        except Exception as e:
            errors.append(f"{uploaded.name}: {e}")
            markdown_parts.append(f"# {uploaded.name}\n\n*Erreur de conversion: {e}*")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    progress.progress(1.0, text="Termine !")

    # Combine all parts
    combined_md = "\n\n---\n\n".join(markdown_parts)

    if errors:
        with st.expander(f"{len(errors)} erreur(s) de conversion"):
            for err in errors:
                st.warning(err)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — Result
    # ═════════════════════════════════════════════════════════════════════════

    st.markdown(
        '<div class="step-header"><span class="step-num">3</span> Resultat</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="stats">
      <span class="stat"><b>{len(uploaded_files)}</b> PDF{"s" if len(uploaded_files) > 1 else ""} converties</span>
      <span class="stat"><b>{len(combined_md):,}</b> caracteres</span>
      <span class="stat"><b>{len(errors)}</b> erreur{"s" if len(errors) != 1 else ""}</span>
    </div>
    """, unsafe_allow_html=True)

    # Preview
    with st.expander("Apercu du Markdown", expanded=True):
        preview = combined_md[:5000]
        if len(combined_md) > 5000:
            preview += f"\n\n... *({len(combined_md) - 5000:,} caracteres restants)*"
        st.markdown(preview)

    # Download
    st.download_button(
        "Telecharger le Markdown",
        data=combined_md,
        file_name="converted.md",
        mime="text/markdown",
    )
