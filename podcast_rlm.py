"""
Lex Fridman podcast demo with DSPy RLM.

Inspired by: https://github.com/avbiswas/fast-rlm/blob/main/examples/podcast.py
Paper: "Recursive Language Models" (Zhang et al., 2025, arXiv:2512.24601)

The CSV has 319 episodes (~37M chars) — far beyond any single LLM context window.
DSPy's RLM loads it as a Python variable in a REPL and lets the model write code
to explore, filter, and summarise iteratively.

Run:
  python podcast_rlm.py  (OPENAI_API_KEY is loaded from .env)

Download data first if needed:
  python download_data.py
"""

from dotenv import load_dotenv
load_dotenv()

import dspy

# --- Load CSV as plain text (RLM will write code to parse it) ---
DATA_FILE = "data/lex_fridman_dataset.csv"

try:
    with open(DATA_FILE, encoding="utf-8", errors="replace") as f:
        csv_text = f.read()
except FileNotFoundError:
    raise SystemExit(f"Run `python download_data.py` first to get {DATA_FILE}")

print(f"Loaded {len(csv_text):,} chars from {DATA_FILE}")

# --- Configure DSPy ---
dspy.configure(lm=dspy.LM("openai/gpt-4o"))

# --- Build context (query at top so RLM sees it before the data) ---
context = f"""The following is a CSV file with transcripts from the Lex Fridman Podcast.
Columns: id, guest, title, text

CSV data starts below:
{csv_text}"""

query = (
    "Find what the first 5 Machine Learning guests had to say about AGI "
    "in the Lex Fridman Podcast. Not all guests are ML guests — focus on "
    "established researchers known for their contribution to AI/ML. "
    "Return a summary per guest about what they said about AGI."
)

print(f"\nQuery: {query}\n")

# --- Run RLM ---
rlm = dspy.RLM("context, query -> answer")

result = rlm(context=context, query=query)

print("\n=== Result ===")
print(result.answer)
