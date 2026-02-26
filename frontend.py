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

# --- Load CSV once (cached across reruns) ---
@st.cache_resource(show_spinner="Loading transcripts…")
def load_data():
    with open(DATA_FILE, encoding="utf-8", errors="replace") as f:
        csv_text = f.read()
    context = (
        "The following is a CSV file with transcripts from the Lex Fridman Podcast.\n"
        "Columns: id, guest, title, text\n\n"
        "CSV data starts below:\n"
        + csv_text
    )
    return context

@st.cache_resource(show_spinner="Configuring DSPy…")
def load_rlm():
    dspy.configure(lm=dspy.LM("openai/gpt-4o"))
    return dspy.RLM("context, query -> answer")

# --- UI ---
st.title("Lex Fridman Podcast — RLM Demo")
st.caption("Recursive Language Model (DSPy) over 319 episodes (~37M chars)")

query = st.text_area("Query", value=DEFAULT_QUERY, height=120)

if st.button("Run RLM", type="primary", disabled=not query.strip()):
    try:
        context = load_data()
        rlm = load_rlm()
    except FileNotFoundError:
        st.error(f"`{DATA_FILE}` not found — run `python download_data.py` first.")
        st.stop()

    with st.spinner("Running RLM… (may take a few minutes)"):
        result = rlm(context=context, query=query)

    st.subheader("Result")
    st.markdown(result.answer)
