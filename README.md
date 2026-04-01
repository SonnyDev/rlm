# RLM — Recursive Language Models

Implémentations du pattern Recursive Language Model (RLM) sur différents cas d'usage.

> **Paper** : *Recursive Language Models* (Zhang et al., 2025, arXiv:2512.24601) — inclus dans `RLM.pdf`

## Principe

Un RLM donne à un LLM l'accès à un REPL Python. Le modèle écrit du code, voit le résultat, et itère. Pour les tâches sémantiques (extraction, classification, résumé), il délègue à `llm_query()` — un sous-appel LLM. Cela permet d'analyser des corpus qui dépassent largement la fenêtre de contexte d'un LLM standard.

```
Contexte (ex: 37M chars)
    |
    v
LLM écrit du Python dans un REPL
    |-- llm_query(prompt)        <- sous-appel sémantique
    |-- llm_query_batched([...]) <- sous-appels parallèles
    |-- print(result)            <- observe le résultat
    |-- ... itère ...
    v
FINAL(answer)
```

## Implémentations

| Dossier | Cas d'usage | Backend | Données |
|---------|------------|---------|---------|
| [Racine](#podcast-lex-fridman--dspy) | Podcast Lex Fridman | DSPy + OpenAI | 319 épisodes, 37M chars |
| [rlm_adk/](rlm_adk/) | Podcast Lex Fridman | Google ADK + Gemini | idem |
| [rlm_cours_de_comptes/](rlm_cours_de_comptes/) | Recommandations Cour des Comptes | DSPy + OpenAI/Gemini | 507 rapports HTML, data.gouv.fr |

## Podcast Lex Fridman — DSPy

Démo de référence : analyse de 319 transcriptions de podcasts (37M caractères).

```bash
# Télécharger les données
python download_data.py

# Streamlit (interactif, avec trajectoire)
rlm_env/bin/streamlit run frontend.py

# Script CLI
rlm_env/bin/python podcast_rlm.py
```

Fichiers :
- `frontend.py` — Streamlit avec `InstrumentedRLM`, trajectoire REPL, sous-appels tracés
- `podcast_rlm.py` — Script CLI standalone
- `download_data.py` — Téléchargement du dataset Kaggle

## Setup

```bash
python -m venv rlm_env
source rlm_env/bin/activate
pip install -r requirements.txt
```

Variables d'environnement dans `.env` :
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```
