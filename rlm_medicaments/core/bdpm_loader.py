"""Loading and parsing of BDPM (Base de Données Publique des Médicaments) data."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path


@dataclass
class Medication:
    cis: str
    denomination: str
    forme_pharmaceutique: str = ""
    voies_administration: str = ""
    statut_amm: str = ""
    type_procedure: str = ""
    etat_commercialisation: str = ""
    date_amm: str = ""
    titulaire: str = ""


@dataclass
class Substance:
    cis: str
    designation_element: str = ""
    code_substance: str = ""
    denomination_substance: str = ""
    dosage: str = ""
    reference_dosage: str = ""
    nature_composant: str = ""  # SA, FT


@dataclass
class PairCandidate:
    med_a: Medication
    med_b: Medication


@dataclass
class InteractionResult:
    med_a: Medication
    med_b: Medication
    risk_level: str = ""  # faible, modéré, élevé, critique
    profile_a: str = ""
    profile_b: str = ""
    justification: str = ""
    references: str = ""


def _parse_tsv(filepath: Path, encoding: str = "latin-1") -> list[list[str]]:
    """Parse a tab-separated BDPM file (.txt or .csv, both use tab delimiter)."""
    rows = []
    text = filepath.read_text(encoding=encoding, errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if row and any(cell.strip() for cell in row):
            rows.append(row)
    return rows


def load_medications(filepath: Path) -> list[Medication]:
    """Parse CIS_bdpm.txt → list of Medication."""
    rows = _parse_tsv(filepath)
    meds = []
    for row in rows:
        if len(row) < 2:
            continue
        med = Medication(
            cis=row[0].strip(),
            denomination=row[1].strip(),
            forme_pharmaceutique=row[2].strip() if len(row) > 2 else "",
            voies_administration=row[3].strip() if len(row) > 3 else "",
            statut_amm=row[4].strip() if len(row) > 4 else "",
            type_procedure=row[5].strip() if len(row) > 5 else "",
            etat_commercialisation=row[6].strip() if len(row) > 6 else "",
            date_amm=row[7].strip() if len(row) > 7 else "",
            titulaire=row[10].strip() if len(row) > 10 else "",
        )
        meds.append(med)
    return meds


def load_compositions(filepath: Path) -> dict[str, list[Substance]]:
    """Parse CIS_COMPO_bdpm.txt → dict CIS → list of Substance."""
    rows = _parse_tsv(filepath)
    compo: dict[str, list[Substance]] = {}
    for row in rows:
        if len(row) < 4:
            continue
        sub = Substance(
            cis=row[0].strip(),
            designation_element=row[1].strip() if len(row) > 1 else "",
            code_substance=row[2].strip() if len(row) > 2 else "",
            denomination_substance=row[3].strip() if len(row) > 3 else "",
            dosage=row[4].strip() if len(row) > 4 else "",
            reference_dosage=row[5].strip() if len(row) > 5 else "",
            nature_composant=row[6].strip() if len(row) > 6 else "",
        )
        compo.setdefault(sub.cis, []).append(sub)
    return compo


def get_active_substances(cis: str, compositions: dict[str, list[Substance]]) -> set[str]:
    """Return the set of active substance codes for a medication."""
    subs = compositions.get(cis, [])
    return {s.code_substance for s in subs if s.nature_composant in ("SA", "")}


def filter_medications(
    medications: list[Medication],
    search_term: str = "",
    commercialise_only: bool = True,
) -> list[Medication]:
    """Filter medications by search term and commercialization status."""
    results = medications
    if commercialise_only:
        results = [m for m in results if "commercialis" in m.etat_commercialisation.lower()]
    if search_term:
        term = search_term.lower()
        results = [m for m in results if term in m.denomination.lower()]
    return results


def generate_candidate_pairs(
    medications: list[Medication],
    compositions: dict[str, list[Substance]],
    known_interactions: set[tuple[str, str]] | None = None,
) -> list[PairCandidate]:
    """Generate all candidate pairs, excluding same-substance and known interactions."""
    if known_interactions is None:
        known_interactions = set()

    pairs = []
    for med_a, med_b in combinations(medications, 2):
        # Exclude pairs sharing an active substance
        subs_a = get_active_substances(med_a.cis, compositions)
        subs_b = get_active_substances(med_b.cis, compositions)
        if subs_a & subs_b:
            continue

        # Exclude known interactions (check both orderings)
        key_ab = (med_a.cis, med_b.cis)
        key_ba = (med_b.cis, med_a.cis)
        if key_ab in known_interactions or key_ba in known_interactions:
            continue

        pairs.append(PairCandidate(med_a=med_a, med_b=med_b))

    return pairs


def build_rcp_context(med: Medication, compositions: dict[str, list[Substance]]) -> str:
    """Build a textual context for a medication from available BDPM data.

    Contains structured composition data. Mechanistic profile must be obtained
    via llm_query() using pharmacological knowledge.
    """
    lines = [
        f"MEDICAMENT : {med.denomination}",
        f"Code CIS : {med.cis}",
        f"Forme : {med.forme_pharmaceutique}",
        f"Voie : {med.voies_administration}",
        f"Titulaire : {med.titulaire}",
        "COMPOSITION :",
    ]
    subs = compositions.get(med.cis, [])
    for s in subs:
        nature = "substance active" if s.nature_composant == "SA" else s.nature_composant
        lines.append(f"  - {s.denomination_substance} {s.dosage} ({nature})")
    return "\n".join(lines)
