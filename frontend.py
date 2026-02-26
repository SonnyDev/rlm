import streamlit as st
from dotenv import load_dotenv
import dspy

load_dotenv()

DATA_FILE = "data/lex_fridman_dataset.csv"
DEFAULT_QUERY = (
    "Find what the first 5 Machine Learning guests had to say about AGI "
    "in the Lex Fridman Podcast. Not all guests are ML guests — focus on "
    "established researchers known for their contribution to AI/ML. "
    "Return a summary per guest about what they said about AGI."
)

st.set_page_config(
    page_title="Lex Fridman RLM Demo",
    page_icon="🎙️",
    layout="centered",
)

st.markdown("""
<style>
  /* Page background */
  .stApp { background-color: #0e0e0e; color: #f0f0f0; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Hero */
  .hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
  }
  .hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #ffffff;
    margin-bottom: 0.3rem;
  }
  .hero p {
    color: #888;
    font-size: 0.95rem;
  }

  /* Stat pills */
  .stats {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin: 1.2rem 0 2rem;
    flex-wrap: wrap;
  }
  .stat {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-size: 0.82rem;
    color: #aaa;
  }
  .stat b { color: #fff; }

  /* Text area override */
  .stTextArea textarea {
    background: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 10px !important;
    color: #f0f0f0 !important;
    font-size: 0.92rem !important;
    padding: 0.75rem !important;
  }
  .stTextArea textarea:focus {
    border-color: #555 !important;
    box-shadow: none !important;
  }

  /* Button */
  .stButton > button {
    width: 100%;
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem !important;
    margin-top: 0.5rem;
    transition: opacity 0.15s;
  }
  .stButton > button:hover { opacity: 0.85; }
  .stButton > button:disabled { opacity: 0.3 !important; }

  /* Result card */
  .result-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    line-height: 1.7;
    font-size: 0.93rem;
    color: #ddd;
  }
  .result-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #555;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
  }

  /* Spinner text */
  .stSpinner > div { color: #888 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading 319 transcripts…")
def load_data():
    with open(DATA_FILE, encoding="utf-8", errors="replace") as f:
        csv_text = f.read()
    return (
        "The following is a CSV file with transcripts from the Lex Fridman Podcast.\n"
        "Columns: id, guest, title, text\n\n"
        "CSV data starts below:\n"
        + csv_text
    )

@st.cache_resource(show_spinner="Configuring model…")
def load_rlm():
    dspy.configure(lm=dspy.LM("openai/gpt-4o"))
    return dspy.RLM("context, query -> answer")


# --- Hero ---
st.markdown("""
<div class="hero">
  <h1>🎙️ Lex Fridman RLM Demo</h1>
  <p>Recursive Language Model querying 319 podcast transcripts</p>
</div>
<div class="stats">
  <span class="stat"><b>319</b> episodes</span>
  <span class="stat"><b>37M</b> chars</span>
  <span class="stat"><b>DSPy</b> RLM</span>
  <span class="stat"><b>GPT-4o</b></span>
</div>
""", unsafe_allow_html=True)

# --- Query input ---
query = st.text_area("", placeholder="Ask anything about the podcast…",
                     value=DEFAULT_QUERY, height=130, label_visibility="collapsed")

run = st.button("Run", disabled=not query.strip())

# --- Run ---
if run:
    try:
        context = load_data()
        rlm = load_rlm()
    except FileNotFoundError:
        st.error(f"`{DATA_FILE}` introuvable — lance `python download_data.py` d'abord.")
        st.stop()

    with st.spinner("RLM en cours… (quelques minutes)"):
        result = rlm(context=context, query=query)

    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Résultat</div>
      {result.answer.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
