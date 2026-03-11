from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class SubCall:
    prompt: str
    response: str


@dataclass
class RLMIteration:
    step: int
    code: str = ""
    output: str = ""
    reasoning: str = ""
    subcalls: list[SubCall] = field(default_factory=list)


@dataclass
class RunConfig:
    model: str = ""
    sub_model: str = ""
    max_iterations: int = 0
    max_llm_calls: int = 0
    max_output_chars: int = 0
    query: str = ""


@dataclass
class RLMResult:
    answer: str
    trajectory: list[RLMIteration] = field(default_factory=list)
    total_iterations: int = 0
    total_subcalls: int = 0
    config: RunConfig | None = None

    def export_markdown(self, path: str | Path) -> Path:
        """Export the full run to a readable Markdown file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"# RLM Run — {ts}\n")

        # ── Config ──
        if self.config:
            c = self.config
            lines.append("## Configuration\n")
            lines.append(f"| Parameter | Value |")
            lines.append(f"|---|---|")
            lines.append(f"| Primary model | `{c.model}` |")
            lines.append(f"| Sub model | `{c.sub_model}` |")
            lines.append(f"| Max iterations | {c.max_iterations} |")
            lines.append(f"| Max LLM calls | {c.max_llm_calls} |")
            lines.append(f"| Truncate length | {c.max_output_chars:,} chars |")
            lines.append("")
            lines.append(f"**Query:** {c.query}\n")

        # ── Summary ──
        lines.append("## Summary\n")
        lines.append(f"- **Iterations:** {self.total_iterations}")
        lines.append(f"- **Sub-LLM calls:** {self.total_subcalls}\n")

        # ── Trajectory ──
        lines.append("## Trajectory\n")
        n = len(self.trajectory)
        for it in self.trajectory:
            is_last = it.step == n
            tag = " ✅ FINAL" if is_last else ""
            llm_tag = f" — {len(it.subcalls)} sub-LLM call(s)" if it.subcalls else ""
            lines.append(f"### Step {it.step}/{n}{tag}{llm_tag}\n")

            if it.reasoning:
                lines.append(f"> {it.reasoning}\n")

            if it.code:
                lines.append("```python")
                lines.append(it.code)
                lines.append("```\n")

            for j, sc in enumerate(it.subcalls, 1):
                lines.append(f"<details><summary>Sub-LLM call #{j}</summary>\n")
                lines.append(f"**Prompt:**\n```\n{sc.prompt}\n```\n")
                lines.append(f"**Response:**\n```\n{sc.response}\n```\n")
                lines.append("</details>\n")

            if it.output:
                lines.append("**Output:**")
                lines.append(f"```\n{it.output}\n```\n")

            lines.append("---\n")

        # ── Final answer ──
        lines.append("## Final Answer\n")
        lines.append(self.answer)
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path
