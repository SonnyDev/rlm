# Spécifications — Pipeline B3
## Détection d'Interactions Médicamenteuses Implicites Non Documentées
> Pipeline RLM sur données data.gouv.fr  
> Stack cible : DSPy · Streamlit · MCP data.gouv.fr · MarkItDown

---

## 1. Contexte & Objectif

### Problème adressé

La Base de Données Publique des Médicaments (BDPM) publie une liste d'interactions officiellement déclarées entre médicaments. Cette liste est construite à partir de déclarations volontaires et d'études spécifiques. Elle est donc nécessairement incomplète.

Des interactions cliniquement significatives peuvent exister entre deux médicaments sans jamais avoir été formellement étudiées ni déclarées, alors que leurs mécanismes d'action respectifs — décrits en texte libre dans leurs Résumés des Caractéristiques Produits (RCP) — suggèrent une incompatibilité.

### Question centrale

> *"Quels médicaments ont des mécanismes d'action décrits dans leurs RCP qui suggèrent une interaction potentiellement grave, sans qu'aucune interaction officielle ne soit déclarée entre eux dans la BDPM ?"*

### Pourquoi ni SQL ni RAG ne suffisent

**SQL** ne peut pas lire le texte libre des RCP. Les mécanismes d'action ne sont pas des champs structurés.

**RAG** retourne des RCP similaires thématiquement. Il est structurellement incapable de comparer deux RCP *différents* pour y détecter une incompatibilité mécanistique implicite. Il cherche de la ressemblance, pas de la contradiction entre deux documents.

**LLM standard** peut analyser une paire de RCP, mais ne peut pas traverser l'ensemble des 15 000 notices en maintenant un état global cohérent dans une seule fenêtre de contexte.

**RLM** lit chaque RCP en entier via des sous-appels dédiés, extrait les profils mécanistiques, et les confronte dans un appel racine — de façon systématique sur l'ensemble du corpus.

---

## 2. Dataset

Un seul dataset est nécessaire pour cette démonstration.

### Base de Données Publique des Médicaments — BDPM
- **Producteur :** ANSM
- **URL data.gouv.fr :** `datasets/base-de-donnees-publique-des-medicaments-base-officielle`
- **Accès :** Via le MCP officiel data.gouv.fr (`https://mcp.data.gouv.fr/mcp`)
- **Fichiers utilisés :**
  - **RCP (CIS_RCP.zip)** — Résumés des Caractéristiques Produits, texte intégral. Environ 15 000 médicaments, de 5 à 60 pages par notice. C'est la source principale du pipeline.
  - **Référentiel médicaments (CIS_bdpm.txt)** — Code CIS, dénomination, laboratoire, statut AMM. Utilisé pour identifier et filtrer les médicaments à analyser.
  - **Composition (CIS_COMPO_bdpm.txt)** — Substances actives et dosages. Utilisé pour identifier les paires partageant une même substance active (à exclure de l'analyse).
  - **Interactions déclarées (CIS_INTERACTIONS.txt)** — Liste officielle des interactions connues. Sert de référence négative : les paires déjà présentes dans ce fichier sont exclues des résultats.
- **Format :** CSV tabulé + ZIP contenant les RCP en texte brut
- **Mise à jour :** Hebdomadaire

### Note sur MarkItDown
Les RCP de la BDPM sont disponibles en texte brut — MarkItDown n'est **pas nécessaire** pour ce pipeline.

---

## 3. Périmètre de la Démonstration

Pour une démonstration à temps limité, il est recommandé de restreindre le corpus à une classe ATC unique plutôt que de lancer le pipeline sur les 15 000 médicaments.

**Classe ATC suggérée pour la démo :** Système cardiovasculaire (classe C) ou anticoagulants (B01) — classes thérapeutiques riches en interactions connues et inconnues, avec un fort enjeu clinique.

Ce périmètre réduit permet de valider la chaîne technique complète sur un corpus de quelques centaines de médicaments et quelques milliers de paires, tout en produisant des résultats cliniquement interprétables.

---

## 4. Architecture du Pipeline

### 4.1 Rôle de chaque composant

**MCP data.gouv.fr** — téléchargement et parsing des ressources BDPM directement via l'API MCP officielle, sans pipeline d'ingestion custom.

**DSPy** — orchestration des modules RLM : définition des signatures d'entrée/sortie, enchaînement des sous-appels, agrégation des résultats.

**Streamlit** — interface de lancement, paramétrage du périmètre (classe ATC, seuil de risque), visualisation des résultats et export du rapport.

### 4.2 Structure de répertoire suggérée

```
project/
├── main.py              → Application Streamlit
├── core/
│   ├── mcp_client.py    → Wrapper MCP data.gouv.fr
│   ├── bdpm_loader.py   → Chargement et parsing BDPM
│   └── rlm_pipeline.py  → Module RLM DSPy
└── outputs/
    └── b3_results.json  → Rapport produit par le pipeline
```

---

## 5. Logique de Traitement RLM

### 5.1 Constitution des paires candidates

Avant d'appeler le RLM, construire la liste des paires de médicaments à analyser :

- Prendre tous les médicaments du périmètre retenu (classe ATC choisie)
- Générer toutes les paires possibles au sein de ce périmètre
- Exclure les paires dont les deux médicaments partagent la même substance active
- Exclure les paires déjà présentes dans le fichier d'interactions officielles de la BDPM

Les paires restantes constituent le corpus à soumettre au RLM.

### 5.2 Traitement RLM par paire

Pour chaque paire candidate (médicament A, médicament B), le RLM effectue trois appels :

**Sous-appel 1 — Lecture du RCP du médicament A**
Le modèle lit l'intégralité du RCP du médicament A et en extrait le profil mécanistique : mécanisme d'action principal, voies métaboliques impliquées (enzymes CYP450, transporteurs membranaires), cibles pharmacologiques, organes principalement affectés, et tout effet sur l'homéostasie (électrolytes, coagulation, pression artérielle, etc.).

**Sous-appel 2 — Lecture du RCP du médicament B**
Même extraction sur le RCP du médicament B.

**Appel racine — Confrontation des deux profils**
Le modèle reçoit les deux profils mécanistiques extraits et raisonne sur leur compatibilité. Il évalue si les mécanismes décrits suggèrent une interaction potentiellement cliniquement significative qui ne serait pas documentée dans la BDPM. Il produit un niveau de risque et une justification textuelle.

### 5.3 Profondeur de récursion
La profondeur est fixée à **1 niveau** pour cette démonstration : un appel racine par paire, avec deux sous-appels (un par RCP). Aucune récursion supplémentaire n'est nécessaire pour ce cas d'usage.

---

## 6. Outputs

### 6.1 Format du résultat par paire signalée

Pour chaque paire pour laquelle le RLM détecte une interaction potentielle :

- **Identification** — noms et codes CIS des deux médicaments, laboratoires, classes ATC
- **Statut BDPM** — confirmation que la paire est absente du fichier d'interactions officielles
- **Niveau de risque estimé** — faible / modéré / élevé / critique
- **Profil mécanistique du médicament A** — extrait de la justification du sous-appel 1
- **Profil mécanistique du médicament B** — extrait de la justification du sous-appel 2
- **Justification de l'interaction implicite** — raisonnement produit par l'appel racine
- **Références sources** — sections des RCP ayant fondé le raisonnement (ex. : "Section 5.1 du RCP — Propriétés pharmacodynamiques")

### 6.2 Rapport de synthèse

En fin d'exécution, le pipeline produit un rapport agrégé indiquant :
- Nombre total de paires analysées
- Nombre de paires signalées par niveau de risque
- Répartition par classe ATC et par laboratoire
- Liste triée par niveau de risque décroissant

### 6.3 Avertissement obligatoire sur la portée des résultats
Les résultats produits sont des **hypothèses à valider par des professionnels de santé**. Ils ne constituent pas un avis médical ou réglementaire. Ils doivent être présentés comme des signaux d'alerte nécessitant une revue experte avant toute conclusion.

---

## 7. Paramètres Configurables dans Streamlit

- **Classe ATC** — périmètre des médicaments analysés
- **Niveau de risque minimum** — ne remonter que les paires au-dessus d'un seuil (ex. : modéré et au-dessus)
- **Choix du sous-LLM** — modèle utilisé pour les sous-appels (peut différer du modèle racine)
- **Export** — téléchargement du rapport en JSON ou CSV

---

## 8. Limites Connues

**Coût d'inférence** — même restreint à une classe ATC, le nombre de paires peut être important. Prévoir un mécanisme de cache pour ne pas re-analyser les paires déjà traitées lors d'une nouvelle exécution.

**Faux positifs** — le RLM peut signaler des interactions théoriquement plausibles mais cliniquement non significatives aux doses thérapeutiques usuelles. La revue humaine est indispensable.

**Qualité des RCP** — certains RCP sont plus détaillés que d'autres sur les mécanismes d'action. Un RCP peu informatif produira une extraction pauvre et pourrait manquer une interaction réelle.