# RLM ADK — Podcast Lex Fridman (Google ADK + Gemini)

Variante de la demo podcast utilisant Google ADK (Agent Development Kit) et les modeles Gemini au lieu de DSPy + OpenAI.

## Architecture

```
Lex Fridman CSV (319 episodes, 37M chars)
    |
    v
RLMAgent(BaseAgent)          <- agent.py
    |-- Gemini ecrit du Python
    |-- llm_query() via google.genai
    |-- llm_query_batched() concurrent (ThreadPoolExecutor)
    |-- REPL sandbox           <- repl.py
    v
RLMResult + export Markdown   <- types.py
```

## Fichiers

| Fichier | Role |
|---------|------|
| `agent.py` | `RLMAgent(BaseAgent)` — boucle RLM avec Gemini |
| `repl.py` | REPL Python sandboxe avec `llm_query`, `FINAL` |
| `prompts.py` | Prompt systeme RLM + builders |
| `types.py` | `RLMResult`, `RLMIteration`, `SubCall`, export Markdown |
| `frontend.py` | Streamlit avec trajectoire REPL |

## Lancement

```bash
cd /chemin/vers/rlm
rlm_env/bin/streamlit run rlm_adk/frontend.py
```

Necessite `GOOGLE_API_KEY` dans `.env`.

## Modeles supportes

- `gemini-2.5-flash` (defaut)
- `gemini-2.5-pro`
- `gemini-2.0-flash`
