"""RLM pipeline pour le suivi des recommandations de la Cour des Comptes.

Utilise dspy.RLM avec InstrumentedRLM — le LLM ecrit du code Python dans un REPL,
appelle llm_query() pour analyser chaque rapport individuellement (niveau 1),
puis agrege les resultats pour detecter les recurrences (niveau 2).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import dspy


# ── InstrumentedRLM ──────────────────────────────────────────────────────────


class InstrumentedRLM(dspy.RLM):
    """RLM that records every llm_query call per iteration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subcalls: dict[int, list[dict]] = {}
        self._iter_subcalls: list[dict] = []
        self._lock = threading.Lock()

    def _make_llm_tools(self, max_workers: int = 8):
        tools = super()._make_llm_tools(max_workers)
        orig_query = tools["llm_query"]
        orig_batched = tools["llm_query_batched"]

        def llm_query(prompt: str) -> str:
            result = orig_query(prompt)
            with self._lock:
                self._iter_subcalls.append({"prompt": prompt, "response": result})
            return result

        def llm_query_batched(prompts: list) -> list:
            results = orig_batched(prompts)
            with self._lock:
                for p, r in zip(prompts, results):
                    self._iter_subcalls.append({"prompt": p, "response": r})
            return results

        tools["llm_query"] = llm_query
        tools["llm_query_batched"] = llm_query_batched
        return tools

    def _execute_iteration(self, repl, variables, history, iteration, input_args, output_field_names):
        with self._lock:
            self._iter_subcalls = []
        result = super()._execute_iteration(repl, variables, history, iteration, input_args, output_field_names)
        with self._lock:
            if self._iter_subcalls:
                self.subcalls[iteration] = list(self._iter_subcalls)
        return result

    def forward(self, **input_args):
        self.subcalls.clear()
        return super().forward(**input_args)


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    primary_model: str = "openai/gpt-4o"
    sub_model: str = "openai/gpt-4o-mini"
    max_iterations: int = 25
    max_llm_calls: int = 80
    max_output_chars: int = 100_000


DEFAULT_QUERY = (
    "Sur la periode 2013-2016, quelles recommandations de la Cour des Comptes "
    "concernant l'hôpital public ont été formulées à plusieurs reprises dans des "
    "rapports successifs, signalant qu'elles n'ont toujours pas été mises en oeuvre ?"
)

SUGGESTED_QUERIES = [
    # ── Rapides (peu de sous-appels) ──
    (
        "3 recommandations récurrentes",
        "Dans les 5 premiers rapports du corpus, trouve 3 recommandations qui "
        "apparaissent dans au moins 2 rapports différents. Pour chacune, donne "
        "le texte de la recommandation, les rapports où elle apparaît, et si elle "
        "semble avoir été suivie d'effet."
    ),
    (
        "Comparaison de deux rapports",
        "Prends les 2 rapports les plus longs du corpus. Extrais les recommandations "
        "de chacun, puis identifie celles qui traitent du même sujet. Y a-t-il des "
        "problèmes signalés dans les deux rapports ?"
    ),
    (
        "Top 5 recommandations",
        "Extrais les 5 recommandations les plus importantes du corpus (celles qui "
        "semblent les plus urgentes ou structurantes selon le ton du rapport). "
        "Pour chacune, indique le rapport source et l'entité visée."
    ),
    # ── Exhaustives (corpus complet) ──
    (
        "Recommandations récurrentes (complet)",
        "Sur la periode 2013-2016, quelles recommandations de la Cour des Comptes "
        "concernant l'hôpital public ont été formulées à plusieurs reprises dans des "
        "rapports successifs, signalant qu'elles n'ont toujours pas été mises en oeuvre ?"
    ),
    (
        "Reformulations d'un même problème",
        "Parmi tous les rapports du corpus, identifie les cas où un même problème "
        "structurel est signalé dans plusieurs rapports d'années différentes, même "
        "s'il est formulé différemment à chaque fois. Pour chaque problème récurrent, "
        "donne la première et la dernière occurrence, et le nombre de rapports concernés."
    ),
    (
        "Évolution thématique",
        "Comment les préoccupations de la Cour des Comptes ont-elles "
        "évolué entre 2015 et 2016 ? Quels nouveaux sujets sont apparus ? Quels sujets "
        "anciens persistent ?"
    ),
]


class AnalyseCorpusCourDesComptes(dspy.Signature):
    """Tu es un analyste expert des rapports de la Cour des Comptes.
    Tu reponds TOUJOURS en francais.
    Tu analyses un corpus de rapports publics pour repondre a la question posee.
    Tes reponses doivent etre detaillees, structurees et en langage naturel.
    Quand tu utilises llm_query(), formule tes prompts en francais."""

    context: str = dspy.InputField(desc="Corpus de rapports de la Cour des Comptes")
    query: str = dspy.InputField(desc="Question d'analyse posee par l'utilisateur")
    answer: str = dspy.OutputField(desc="Rapport d'analyse detaille en francais")


def build_rlm(config: PipelineConfig) -> InstrumentedRLM:
    """Create an InstrumentedRLM instance."""
    return InstrumentedRLM(
        AnalyseCorpusCourDesComptes,
        max_iterations=config.max_iterations,
        max_llm_calls=config.max_llm_calls,
        max_output_chars=config.max_output_chars,
        sub_lm=dspy.LM(config.sub_model),
    )
