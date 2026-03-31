"""Chargement et filtrage du corpus de rapports de la Cour des Comptes."""

from __future__ import annotations


# Themes and associated keywords for filtering reports
THEMES = {
    "hopital public": [
        "hopital", "hôpital", "hospitalier", "hospitalière", "CHU", "AP-HP",
        "urgences", "établissement de santé", "établissements de santé",
        "soins hospitaliers", "offre de soins", "réforme hospitalière",
        "groupement hospitalier", "GHT",
    ],
    "education nationale": [
        "éducation nationale", "enseignement scolaire", "école", "lycée",
        "collège", "enseignant", "professeur", "académie", "rectorat",
        "élève", "scolarité",
    ],
    "securite sociale": [
        "sécurité sociale", "assurance maladie", "CNAM", "RALFSS",
        "branche maladie", "financement de la sécurité sociale",
        "déficit social", "ONDAM",
    ],
    "collectivites territoriales": [
        "collectivité", "collectivités territoriales", "commune",
        "département", "région", "intercommunalité", "décentralisation",
        "finances locales",
    ],
    "defense": [
        "défense", "armées", "militaire", "ministère des armées",
        "équipement militaire", "opérations extérieures", "OPEX",
    ],
    "tous": [],  # no filter — all reports
}


def filter_reports_by_theme(
    reports: list[dict],
    theme: str,
) -> list[dict]:
    """Filter reports by theme keywords.

    Each report is a dict with at least {"text": str, "year": int, "filename": str}.
    Returns reports whose text contains at least one keyword from the theme.
    """
    keywords = THEMES.get(theme, [])
    if not keywords:
        return reports  # "tous" — return everything

    filtered = []
    for report in reports:
        text_lower = report["text"].lower()
        if any(kw.lower() in text_lower for kw in keywords):
            filtered.append(report)

    return filtered


def filter_reports_by_period(
    reports: list[dict],
    year_min: int,
    year_max: int,
) -> list[dict]:
    """Filter reports by year range (inclusive)."""
    return [r for r in reports if year_min <= r["year"] <= year_max]


def build_rlm_context(reports: list[dict], max_chars_per_report: int = 50_000) -> str:
    """Build the RLM context string from a list of reports.

    Each report is truncated to max_chars_per_report to keep context manageable.
    The full text is available to the RLM for sub-calls.
    """
    parts = []
    for i, report in enumerate(reports, 1):
        text = report["text"]
        truncated = text[:max_chars_per_report]
        if len(text) > max_chars_per_report:
            truncated += f"\n[... TRONQUE a {max_chars_per_report:,} chars sur {len(text):,} au total]"

        parts.append(
            f"=== RAPPORT {i}/{len(reports)} ===\n"
            f"Fichier: {report['filename']}\n"
            f"Annee: {report['year']}\n"
            f"Taille: {len(text):,} caracteres\n\n"
            f"{truncated}"
        )

    header = (
        "CORPUS DE RAPPORTS DE LA COUR DES COMPTES (data.gouv.fr)\n"
        f"Nombre de rapports: {len(reports)}\n"
        f"Annees couvertes: {min(r['year'] for r in reports)}-{max(r['year'] for r in reports)}\n"
        f"Volume total: {sum(len(r['text']) for r in reports):,} caracteres\n\n"
    )

    return header + "\n\n".join(parts)


def get_report_texts(reports: list[dict]) -> dict[str, str]:
    """Return a dict of filename -> full text for all reports."""
    return {r["filename"]: r["text"] for r in reports}


def summarize_corpus(reports: list[dict]) -> dict:
    """Return summary statistics about the corpus."""
    total_chars = sum(len(r["text"]) for r in reports)
    by_year: dict[int, int] = {}
    for r in reports:
        by_year[r["year"]] = by_year.get(r["year"], 0) + 1

    return {
        "total_reports": len(reports),
        "total_chars": total_chars,
        "total_tokens_approx": total_chars // 4,
        "by_year": dict(sorted(by_year.items())),
    }
