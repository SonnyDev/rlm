"""
Download Lex Fridman podcast transcripts for the DSPy RLM demo.

Source: https://www.kaggle.com/datasets/rajneesh231/lex-fridman-podcast-transcript
No authentication required — public dataset.

Saves: data/lex_fridman_dataset.csv
"""

import os
import io
import urllib.request
import zipfile

DATA_DIR = "data"
KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "rajneesh231/lex-fridman-podcast-transcript"
)
OUT_PATH = os.path.join(DATA_DIR, "lex_fridman_dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)

if os.path.exists(OUT_PATH):
    print(f"Already downloaded: {OUT_PATH}")
else:
    print("Downloading Lex Fridman podcast transcripts from Kaggle (~12 MB)…")
    req = urllib.request.Request(KAGGLE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    print(f"  Downloaded {len(data):,} bytes")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        print(f"  Zip contents: {names}")
        csv_name = next((n for n in names if n.endswith(".csv")), names[0])
        with zf.open(csv_name) as src, open(OUT_PATH, "wb") as dst:
            dst.write(src.read())

    print(f"  Saved → {OUT_PATH}")

size = os.path.getsize(OUT_PATH)
print(f"File size: {size:,} bytes ({size // 1024} KB)")

# Quick preview
with open(OUT_PATH, encoding="utf-8", errors="replace") as f:
    head = f.read(500)
print("\nFirst 500 chars:")
print(head)
