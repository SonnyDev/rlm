# RLM ADK — Podcast Lex Fridman (Google ADK + Gemini)

Variante de la démo podcast utilisant Google ADK (Agent Development Kit) et les modèles Gemini au lieu de DSPy + OpenAI.

## Architecture

```
Lex Fridman CSV (319 épisodes, 37M chars)
    |
    v
RLMAgent(BaseAgent)          <- agent.py
    |-- Gemini écrit du Python
    |-- llm_query() via google.genai
    |-- llm_query_batched() concurrent (ThreadPoolExecutor)
    |-- REPL sandbox           <- repl.py
    v
RLMResult + export Markdown   <- types.py
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `agent.py` | `RLMAgent(BaseAgent)` — boucle RLM avec Gemini |
| `repl.py` | REPL Python sandboxé avec `llm_query`, `FINAL` |
| `prompts.py` | Prompt système RLM + builders |
| `types.py` | `RLMResult`, `RLMIteration`, `SubCall`, export Markdown |
| `frontend.py` | Streamlit avec trajectoire REPL |

## Lancement

```bash
cd ..
rlm_env/bin/streamlit run rlm_adk/frontend.py
```

Nécessite `GOOGLE_API_KEY` dans `.env`.

## Modèles supportés

- `gemini-2.5-flash` (défaut)
- `gemini-2.5-pro`
- `gemini-2.0-flash`
