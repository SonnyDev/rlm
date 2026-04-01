# RLM — Recursive Language Models

Implementations du pattern Recursive Language Model (RLM) sur differents cas d'usage.

> **Paper** : *Recursive Language Models* (Zhang et al., 2025, arXiv:2512.24601) — inclus dans `RLM.pdf`

## Principe

Un RLM donne a un LLM l'acces a un REPL Python. Le modele ecrit du code, voit le resultat, et itere. Pour les taches semantiques (extraction, classification, resume), il delegue a `llm_query()` — un sous-appel LLM. Cela permet d'analyser des corpus qui depassent largement la fenetre de contexte d'un LLM standard.

```
Contexte (ex: 37M chars)
    |
    v
LLM ecrit du Python dans un REPL
    |-- llm_query(prompt)        <- sous-appel semantique
    |-- llm_query_batched([...]) <- sous-appels paralleles
    |-- print(result)            <- observe le resultat
    |-- ... itere ...
    v
FINAL(answer)
```

## Implementations

| Dossier | Cas d'usage | Backend | Donnees |
|---------|------------|---------|---------|
| [Racine](#podcast-lex-fridman--dspy) | Podcast Lex Fridman | DSPy + OpenAI | 319 episodes, 37M chars |
| [rlm_adk/](rlm_adk/) | Podcast Lex Fridman | Google ADK + Gemini | idem |
| [rlm_cours_de_comptes/](rlm_cours_de_comptes/) | Recommandations Cour des Comptes | DSPy + OpenAI/Gemini | 507 rapports HTML, data.gouv.fr |
| [rlm_medicaments/](rlm_medicaments/) | Interactions medicamenteuses | DSPy + OpenAI/Gemini | 15 649 medicaments BDPM, data.gouv.fr |

## Podcast Lex Fridman — DSPy

Demo de reference : analyse de 319 transcriptions de podcasts (37M caracteres).

```bash
# Telecharger les donnees
python download_data.py

# Streamlit (interactif, avec trajectoire)
rlm_env/bin/streamlit run frontend.py

# Script CLI
rlm_env/bin/python podcast_rlm.py
```

Fichiers :
- `frontend.py` — Streamlit avec `InstrumentedRLM`, trajectoire REPL, sous-appels traces
- `podcast_rlm.py` — Script CLI standalone
- `download_data.py` — Telechargement du dataset Kaggle

## Setup

```bash
python -m venv rlm_env
source rlm_env/bin/activate
pip install dspy streamlit python-dotenv google-adk google-genai markitdown[pdf]
```

Variables d'environnement dans `.env` :
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```
