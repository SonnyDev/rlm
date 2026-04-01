# RLM Cour des Comptes — Suivi des recommandations récurrentes

Détection des recommandations récurrentes dans les rapports publics de la Cour des Comptes par Recursive Language Model.

## Problème

La Cour des Comptes publie chaque année des rapports contenant des recommandations aux administrations. Certaines recommandations sont reformulées année après année, signalant qu'elles n'ont pas été mises en œuvre. Détecter ces récurrences nécessite de lire et comparer l'intégralité du corpus.

## Architecture

```
data.gouv.fr (MCP)
    |
    v  list_dataset_resources → URLs dynamiques (fallback statique si 502)
    |
    v
4 zips HTML (2013-2016) → 507 rapports extraits en texte
Recommandations TSV     → 1619 recs (vérité terrain 2015-2018)
    |
    v  filtrage par thème + période
    |
InstrumentedRLM (dspy.RLM)
    |
    |-- Niveau 1 : llm_query() par rapport → extraction des recommandations
    |-- Niveau 2 : appel racine → agrégation, détection de récurrence
    v
Streamlit : trajectoire REPL + résultat + export JSON
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `core/mcp_client.py` | Client MCP data.gouv.fr — découverte dynamique + téléchargement |
| `core/corpus_loader.py` | Filtrage par thème (6 thèmes) et période, construction du contexte |
| `core/rlm_pipeline.py` | `InstrumentedRLM` avec signature française, questions suggérées |
| `main.py` | Streamlit : thème, période, questions, trajectoire REPL |

## Lancement

```bash
cd /chemin/vers/rlm
rlm_env/bin/streamlit run rlm_cours_de_comptes/main.py
```

## Thèmes disponibles

- Hôpital public
- Éducation nationale
- Sécurité sociale
- Collectivités territoriales
- Défense
- Tous (sans filtre)

## Données

- **Rapports** : 507 rapports HTML (2013-2016) depuis data.gouv.fr
- **Recommandations** : 1 619 recommandations officielles (2015-2018, vérité terrain)
- **Intersection** : 2015-2016 (période par défaut)

## Modèles supportés

- Gemini : `gemini/gemini-2.5-flash`, `gemini/gemini-2.5-pro`
- OpenAI : `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-5-mini`
