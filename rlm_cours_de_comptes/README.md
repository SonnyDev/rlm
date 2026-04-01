# RLM Cour des Comptes — Suivi des recommandations recurrentes

Detection des recommandations recurrentes dans les rapports publics de la Cour des Comptes par Recursive Language Model.

## Probleme

La Cour des Comptes publie chaque annee des rapports contenant des recommandations aux administrations. Certaines recommandations sont reformulees annee apres annee, signalant qu'elles n'ont pas ete mises en oeuvre. Detecter ces recurrences necessite de lire et comparer l'integralite du corpus — une tache impossible pour un humain ou un LLM classique.

## Architecture

```
data.gouv.fr (MCP)
    |
    v  list_dataset_resources → URLs dynamiques (fallback statique si 502)
    |
    v
4 zips HTML (2013-2016) → 507 rapports extraits en texte
Recommandations TSV     → 1619 recs (verite terrain 2015-2018)
    |
    v  filtrage par theme + periode
    |
InstrumentedRLM (dspy.RLM)
    |
    |-- Niveau 1 : llm_query() par rapport → extraction des recommandations
    |-- Niveau 2 : appel racine → agregation, detection de recurrence
    v
Streamlit : trajectoire REPL + resultat + export JSON
```

## Fichiers

| Fichier | Role |
|---------|------|
| `core/mcp_client.py` | Client MCP data.gouv.fr — decouverte dynamique + telechargement |
| `core/corpus_loader.py` | Filtrage par theme (6 themes) et periode, construction du contexte |
| `core/rlm_pipeline.py` | `InstrumentedRLM` avec signature francaise, questions suggerees |
| `main.py` | Streamlit : theme, periode, questions, trajectoire REPL |

## Lancement

```bash
cd /chemin/vers/rlm
rlm_env/bin/streamlit run rlm_cours_de_comptes/main.py
```

## Themes disponibles

- Hopital public
- Education nationale
- Securite sociale
- Collectivites territoriales
- Defense
- Tous (sans filtre)

## Donnees

- **Rapports** : 507 rapports HTML (2013-2016) depuis data.gouv.fr
- **Recommandations** : 1 619 recommandations officielles (2015-2018, verite terrain)
- **Intersection** : 2015-2016 (periode par defaut)

## Modeles supportes

- Gemini : `gemini/gemini-2.5-flash`, `gemini/gemini-2.5-pro`
- OpenAI : `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-5-mini`
