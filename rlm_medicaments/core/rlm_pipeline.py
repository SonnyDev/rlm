"""RLM pipeline for implicit drug interaction detection using DSPy.

Uses dspy.RLM — the actual Recursive Language Model with REPL execution,
llm_query sub-calls, and full trajectory tracking.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import dspy


# ── InstrumentedRLM ──────────────────────────────────────────────────────────
# Subclass that records every llm_query sub-call per iteration,
# enabling trajectory visualization in the Streamlit UI.


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

RISK_ORDER = {
    "critique": 4,
    "élevé": 3, "eleve": 3,
    "modéré": 2, "modere": 2,
    "faible": 1,
    "aucun": 0,
}


@dataclass
class PipelineConfig:
    primary_model: str = "openai/gpt-4o"
    sub_model: str = "openai/gpt-4o-mini"
    min_risk_level: str = "modere"
    max_iterations: int = 20
    max_llm_calls: int = 50
    max_output_chars: int = 100_000
    user_query: str = ""


class AnalyseInteractionsMedicamenteuses(dspy.Signature):
    """Tu es un pharmacologue expert.
    Tu reponds TOUJOURS en francais.
    Tu analyses des donnees BDPM pour detecter des interactions medicamenteuses implicites.
    Tes reponses doivent etre detaillees, structurees et en langage naturel.
    Quand tu utilises llm_query(), formule tes prompts en francais."""

    context: str = dspy.InputField(desc="Donnees BDPM des medicaments a analyser")
    query: str = dspy.InputField(desc="Question d'analyse posee par l'utilisateur")
    answer: str = dspy.OutputField(desc="Rapport d'analyse detaille en francais")


def build_rlm(config: PipelineConfig) -> InstrumentedRLM:
    """Create an InstrumentedRLM instance with the given config."""
    return InstrumentedRLM(
        AnalyseInteractionsMedicamenteuses,
        max_iterations=config.max_iterations,
        max_llm_calls=config.max_llm_calls,
        max_output_chars=config.max_output_chars,
        sub_lm=dspy.LM(config.sub_model),
    )
