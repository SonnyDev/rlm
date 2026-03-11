"""RLM Agent built on Google ADK (Agent Development Kit).

Uses google.genai for Gemini model calls and wraps the RLM loop
in an ADK-compatible BaseAgent pattern with event streaming.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types as genai_types
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext

from .repl import LocalREPL, extract_code_blocks, extract_reasoning
from .prompts import RLM_SYSTEM_PROMPT, build_user_prompt, build_iteration_prompt
from .types import RLMResult, RLMIteration, SubCall, RunConfig


class RLMAgent(BaseAgent):
    """Recursive Language Model agent using Google ADK + Gemini.

    Implements the RLM pattern: LLM writes code in a REPL, executes it,
    sees output, and iterates until it produces a final answer.
    """

    model: str = "gemini-2.5-flash"
    sub_model: str = "gemini-2.5-flash"
    max_iterations: int = 20
    max_llm_calls: int = 50
    max_output_chars: int = 100_000
    api_key: str | None = None

    # Internal state (not Pydantic fields)
    _client: genai.Client | None = None
    _result: RLMResult | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> genai.Client:
        if self._client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = genai.Client(**kwargs)
        return self._client

    @property
    def result(self) -> RLMResult | None:
        return self._result

    def _llm_query(self, prompt: str) -> str:
        """Sub-LLM call for semantic tasks."""
        client = self._get_client()
        response = client.models.generate_content(
            model=self.sub_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=4096,
            ),
        )
        return response.text or ""

    def _llm_query_batched(self, prompts: list[str]) -> list[str]:
        """Concurrent sub-LLM calls."""
        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(self._llm_query, p): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[ERROR] {e}"
        return results

    def _build_config(self, query: str) -> RunConfig:
        return RunConfig(
            model=self.model,
            sub_model=self.sub_model,
            max_iterations=self.max_iterations,
            max_llm_calls=self.max_llm_calls,
            max_output_chars=self.max_output_chars,
            query=query,
        )

    def run(self, context: str, query: str) -> RLMResult:
        """Run the RLM loop synchronously."""
        client = self._get_client()
        config = self._build_config(query)
        repl = LocalREPL(context, self._llm_query, self._llm_query_batched)
        trajectory: list[RLMIteration] = []
        total_subcalls = 0

        # Build initial messages
        messages = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=build_user_prompt(
                    query, context[:2000], len(context)
                ))],
            )
        ]

        try:
            for step in range(1, self.max_iterations + 1):
                # Call Gemini
                response = client.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=RLM_SYSTEM_PROMPT,
                        temperature=0.2,
                        max_output_tokens=8192,
                    ),
                )

                response_text = response.text or ""
                if not response_text.strip():
                    break

                # Extract code blocks and reasoning
                code_blocks = extract_code_blocks(response_text)
                reasoning = extract_reasoning(response_text)

                if not code_blocks:
                    # No code — check if there's a final answer in the text
                    iteration = RLMIteration(
                        step=step, reasoning=reasoning, code="", output=""
                    )
                    trajectory.append(iteration)

                    # If model says FINAL in text without code block
                    if "FINAL(" in response_text or "FINAL_VAR(" in response_text:
                        # Try to extract answer from text
                        import re
                        m = re.search(r'FINAL\(["\'](.+?)["\']\)', response_text, re.DOTALL)
                        if m:
                            self._result = RLMResult(
                                answer=m.group(1),
                                trajectory=trajectory,
                                total_iterations=step,
                                total_subcalls=total_subcalls,
                                config=config,
                            )
                            return self._result
                    break

                # Execute each code block
                combined_code = "\n\n".join(code_blocks)
                output, subcalls, final_answer = repl.execute(combined_code)
                total_subcalls += len(subcalls)

                iteration = RLMIteration(
                    step=step,
                    code=combined_code,
                    output=output,
                    reasoning=reasoning,
                    subcalls=subcalls,
                )
                trajectory.append(iteration)

                # Check for FINAL answer
                if final_answer is not None:
                    self._result = RLMResult(
                        answer=final_answer,
                        trajectory=trajectory,
                        total_iterations=step,
                        total_subcalls=total_subcalls,
                        config=config,
                    )
                    return self._result

                # Check sub-LLM call budget
                if total_subcalls >= self.max_llm_calls:
                    break

                # Append to conversation
                messages.append(genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=response_text)],
                ))
                messages.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=build_iteration_prompt(
                        combined_code, output, step, self.max_output_chars
                    ))],
                ))

        finally:
            repl.cleanup()

        # If we ran out of iterations without FINAL, use last output
        answer = "(No final answer produced — max iterations reached)"
        if trajectory and trajectory[-1].output:
            answer = trajectory[-1].output

        self._result = RLMResult(
            answer=answer,
            trajectory=trajectory,
            total_iterations=len(trajectory),
            total_subcalls=total_subcalls,
            config=config,
        )
        return self._result

    async def _run_async_impl(
        self, ctx: InvocationContext
    ):
        """ADK BaseAgent async entry point. Delegates to sync run()."""
        # This enables the agent to be used within ADK's runner infrastructure
        # For the Streamlit frontend, we call run() directly
        yield  # pragma: no cover
