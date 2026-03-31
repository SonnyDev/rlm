# Spécifications — Pipeline RLM : Suivi des Recommandations de la Cour des Comptes
> Démonstration de la capacité des RLMs à traiter des millions de tokens là où les LLMs échouent  
> Stack cible : DSPy · Streamlit · MCP data.gouv.fr · MarkItDown

---

## 1. Pourquoi ce cas d'usage illustre les RLMs mieux que tout autre

### La propriété fondamentale des RLMs

Un RLM n'est pas un LLM plus grand. C'est un LLM qui **se décompose récursivement** pour traiter des inputs dont la taille dépasse structurellement sa fenêtre de contexte — et qui maintient un état cohérent sur l'ensemble du corpus traversé.

La démonstration la plus frappante de cette capacité n'est pas un document légèrement trop long. C'est un corpus de **plusieurs dizaines de rapports**, chacun dépassant déjà la fenêtre de contexte, dont la valeur analytique n'émerge qu'en les lisant **tous**, dans l'ordre chronologique, avec une mémoire transversale.

Les rapports de la Cour des Comptes sont exactement ce corpus.

---

### Le problème que les LLMs ne peuvent pas résoudre

La Cour des Comptes publie chaque année entre 20 et 30 rapports thématiques, chacun faisant entre 150 et 500 pages. Sur 10 ans, cela représente **200 à 300 rapports**, soit plusieurs dizaines de millions de tokens au total.

Un LLM avec une fenêtre de 200 000 tokens peut lire **un seul rapport partiellement**. Il ne peut pas :

- Lire l'ensemble des rapports sur un thème donné sur 10 ans
- Détecter qu'une recommandation de 2014 réapparaît reformulée en 2018, 2021 et 2024
- Conclure qu'elle n'a jamais été mise en œuvre
- Maintenir cette mémoire transversale tout au long de la traversée du corpus

Un LLM tenté de répondre à cette question sans RLM produira une réponse partielle basée sur les premiers rapports chargés — **sans signaler qu'il n'a pas lu les suivants**. C'est la failure mode la plus dangereuse : une réponse confidente mais incomplète.

---

## 2. La Question Centrale

> *"Sur les 10 dernières années, quelles recommandations de la Cour des Comptes concernant l'hôpital public ont été formulées à plusieurs reprises dans des rapports successifs, signalant qu'elles n'ont toujours pas été mises en œuvre ?"*

Cette question est **impossible à résoudre** sans lire l'intégralité de tous les rapports pertinents. La réponse n'existe dans aucun document individuel — elle émerge uniquement de la comparaison transversale de l'ensemble du corpus.

---

## 3. Datasets

### 3.1 Rapports publiés par la Cour des Comptes
- **URL data.gouv.fr :** `datasets/rapports-publies-par-la-cour-des-comptes`
- **Producteur :** Cour des Comptes
- **Format :** XML (texte intégral) et HTML
- **Contenu :** Rapports publics thématiques, rapports annuels, rapports au Parlement, RALFSS (Rapport sur l'Application des Lois de Financement de la Sécurité Sociale)
- **Couverture :** Depuis 2016 en open data, avec des extensions jusqu'à 2017-2018 dans des datasets complémentaires
- **Volume :** Plusieurs dizaines de rapports, chacun entre 150 et 500 pages

### 3.2 Recommandations publiées par la Cour des Comptes
- **URL data.gouv.fr :** `datasets/recommandations-publiees-par-la-cour-des-comptes-2015-mai-2018`
- **Producteur :** Cour des Comptes
- **Format :** CSV structuré
- **Contenu :** Recommandations extraites des rapports publics annuels, thématiques et au Parlement entre 2015 et 2018
- **Rôle dans le pipeline :** Dataset de référence pour la période 2015-2018 — permet de valider les extractions RLM sur une vérité terrain partielle

### 3.3 Accès
Les deux datasets sont accessibles via le **MCP officiel data.gouv.fr** (`https://mcp.data.gouv.fr/mcp`) sans restriction.

---

### Note sur MarkItDown

Les rapports de la Cour des Comptes sont disponibles en XML et HTML sur data.gouv.fr — MarkItDown **n'est pas nécessaire** pour ce pipeline. Les fichiers XML sont directement lisibles en texte brut.

MarkItDown ne serait utile que si l'on souhaitait étendre le corpus à des rapports disponibles uniquement en PDF sur le site de la Cour des Comptes hors data.gouv.fr, ce qui sort du périmètre de cette démonstration.

---

## 4. Pourquoi ni SQL ni RAG ni Deep Research ne résolvent ce problème

### SQL
SQL peut filtrer les rapports par thème ou par année si les métadonnées sont structurées. Il ne peut pas lire le contenu des rapports. Il ne peut pas détecter qu'une recommandation de 2016 et une recommandation de 2022 concernent le même problème non résolu, formulé différemment.

### RAG
Le RAG retourne des passages similaires à une requête. Pour détecter une recommandation récurrente, il faudrait savoir exactement quoi chercher — or le signal n'est pas dans un passage isolé mais dans la **répétition à travers le temps**. Le RAG est aveugle à la temporalité et à la récurrence inter-documents.

De plus, le RAG fragmente les rapports en chunks. Une recommandation formulée sur 3 pages dans un rapport de 400 pages ne tient pas dans un chunk de 512 tokens. Le contexte nécessaire à sa compréhension est perdu.

### Deep Research (ChatGPT, Gemini)
Deep Research cherche sur le web par mots-clés. Les fichiers XML bruts de data.gouv.fr ne sont pas des pages web indexées par Google. Deep Research ne peut pas accéder au corpus complet, ni maintenir une mémoire transversale sur 200 rapports, ni détecter une récurrence qui ne se révèle qu'à la lecture exhaustive.

### RLM
Le RLM lit chaque rapport en intégralité via des sous-appels dédiés, extrait les recommandations en maintenant leur contexte complet, et agrège les résultats en maintenant un état global sur l'ensemble du corpus — détectant ainsi les récurrences que ni SQL, ni RAG, ni Deep Research ne peuvent voir.

---

## 5. Architecture du Pipeline

### 5.1 Rôle de chaque composant

**MCP data.gouv.fr** — téléchargement et parsing des rapports XML directement via l'API MCP officielle. Aucun pipeline d'ingestion custom nécessaire.

**DSPy** — orchestration du pipeline RLM : définition des signatures d'extraction, enchaînement des sous-appels par rapport, agrégation transversale.

**Streamlit** — interface de configuration du thème analysé, visualisation du tableau de suivi des recommandations, export du rapport.

### 5.2 Structure de répertoire suggérée

```
project/
├── main.py                  → Application Streamlit
├── core/
│   ├── mcp_client.py        → Wrapper MCP data.gouv.fr
│   ├── corpus_loader.py     → Chargement et filtrage des rapports par thème
│   └── rlm_pipeline.py      → Module RLM DSPy
└── outputs/
    └── suivi_recommandations.json
```

---

## 6. Logique de Traitement RLM

### 6.1 Étape préalable — Constitution du corpus (hors RLM)

Depuis le MCP data.gouv.fr, récupérer la liste de tous les rapports disponibles et les filtrer sur le thème retenu pour la démonstration. Pour la démonstration, le thème retenu est **l'hôpital public** — qui fait l'objet de rapports récurrents de la Cour des Comptes depuis plus de 10 ans.

Le corpus résultant sera une liste ordonnée chronologiquement de rapports, chacun représentant plusieurs centaines de pages, dont le total dépasse structurellement toute fenêtre de contexte de LLM disponible.

### 6.2 Traitement RLM — Niveau 1 : Extraction par rapport

Pour chaque rapport du corpus, un **sous-appel dédié** lit le rapport en intégralité et en extrait :

- Toutes les recommandations formulées, avec leur texte exact
- Le numéro et l'intitulé du chapitre source
- L'entité ou l'administration visée par chaque recommandation
- La nature de la recommandation : organisationnelle, budgétaire, réglementaire, de gouvernance
- Le degré d'urgence signalé par la Cour si explicitement mentionné

Chaque sous-appel produit une liste structurée de recommandations pour son rapport, avec leur contexte complet — pas des chunks isolés.

### 6.3 Traitement RLM — Niveau 2 : Détection de récurrence

L'appel racine reçoit toutes les listes de recommandations extraites au niveau 1, ordonnées chronologiquement, et effectue le raisonnement transversal :

- Regrouper les recommandations traitant du même problème structurel, même si formulées différemment d'un rapport à l'autre
- Pour chaque groupe, identifier la première occurrence, les reformulations successives et la date du rapport le plus récent mentionnant encore le problème
- Évaluer le statut probable de mise en œuvre : une recommandation réapparaissant 3 fois ou plus sur 10 ans est présumée non mise en œuvre
- Identifier les recommandations apparues une seule fois et disparues — potentiellement traitées

### 6.4 Profondeur de récursion

La profondeur est fixée à **2 niveaux** pour cette démonstration :
- Niveau 1 : un sous-appel par rapport (extraction)
- Niveau 2 : un appel racine unique (agrégation transversale)

Cette profondeur est suffisante pour traiter le corpus complet tout en gardant les coûts d'inférence maîtrisables.

---

## 7. Output

### 7.1 Tableau de suivi des recommandations

Pour chaque groupe de recommandations récurrentes identifié :

- **Libellé synthétique** — reformulation unificatrice du problème structurel identifié
- **Première occurrence** — année et rapport source
- **Occurrences successives** — liste chronologique des rapports ayant reformulé la recommandation
- **Dernière occurrence** — année et rapport source le plus récent
- **Nombre total d'occurrences** — indicateur de persistance du problème
- **Nature** — organisationnelle / budgétaire / réglementaire / gouvernance
- **Entité visée** — ministère, agence, établissement
- **Statut présumé** — non mise en œuvre / mise en œuvre partielle / résolu (disparition des occurrences)
- **Extrait représentatif** — citation exacte d'une formulation de la recommandation avec référence au rapport source

### 7.2 Indicateurs de synthèse

- Nombre total de recommandations extraites sur le corpus
- Nombre de groupes récurrents détectés
- Distribution des recommandations par nombre d'occurrences
- Distribution par nature et par entité visée
- Chronologie : quelles années ont produit le plus de nouvelles recommandations vs reformulations d'anciennes

### 7.3 Avertissement sur la portée des résultats

Le pipeline produit une analyse basée sur le texte des rapports publics. Il ne dispose pas d'accès aux réponses des administrations concernées ni aux plans d'action éventuellement mis en place. Le statut "non mise en œuvre" est une inférence basée sur la récurrence textuelle, pas une vérification factuelle de l'état d'avancement réel. Une revue humaine est nécessaire avant toute conclusion définitive.

---

## 8. Validation du Pipeline

### 8.1 Vérité terrain disponible

Le dataset `recommandations-publiees-par-la-cour-des-comptes-2015-mai-2018` fournit une liste structurée des recommandations officiellement extraites par la Cour elle-même sur la période 2015-2018. Il permet de mesurer précisément le taux de rappel du pipeline RLM sur cette période : combien de recommandations officielles le RLM a-t-il retrouvées, et combien a-t-il manquées.

C'est le benchmark de référence pour la démonstration : comparer les recommandations extraites par le RLM avec celles de la vérité terrain officielle sur la période 2015-2018.

### 8.2 Démonstration LLM vs RLM

La démonstration comparative se construit en deux temps :

**LLM standard** — charger les premiers rapports jusqu'à saturation de la fenêtre de contexte et poser la question des recommandations récurrentes. Montrer que la réponse est partielle, limitée aux rapports les plus récents, et que le LLM ne signale pas les rapports qu'il n'a pas lus.

**RLM** — exécuter le pipeline complet sur l'ensemble du corpus. Montrer que des recommandations formulées en 2015 et réapparaissant en 2024 sont détectées, alors qu'elles étaient totalement invisibles pour le LLM standard.

L'écart est mesurable, factuel et directement comparable — c'est la démonstration la plus convaincante de la valeur structurelle des RLMs.

---

## 9. Périmètre de la Démonstration

Pour une démonstration à temps limité, restreindre le corpus au thème **hôpital public** pour les raisons suivantes :

- La Cour des Comptes a publié des rapports sur l'hôpital public de façon récurrente depuis plus de 10 ans — garantissant un corpus suffisamment dense pour que la récurrence soit détectable
- Le sujet est immédiatement compréhensible par n'importe quel audience — tout le monde comprend l'enjeu des réformes hospitalières non mises en œuvre
- Les recommandations sont formulées de façon suffisamment différente d'un rapport à l'autre pour rendre le simple grep inefficace — mais suffisamment proches sémantiquement pour que le RLM les regroupe correctement

---

## 10. Paramètres Configurables dans Streamlit

- **Thème** — sélection du domaine de politique publique à analyser (hôpital public, éducation nationale, défense, collectivités, sécurité sociale)
- **Période** — bornes temporelles du corpus (ex. : 2014-2024)
- **Seuil de récurrence** — nombre minimal d'occurrences pour qu'une recommandation soit considérée comme récurrente (défaut : 2)
- **Nature des recommandations** — filtre par type (budgétaire, organisationnelle, réglementaire)
- **Export** — téléchargement du tableau de suivi en JSON ou CSV