from dataclasses import dataclass, field


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
class RLMResult:
    answer: str
    trajectory: list[RLMIteration] = field(default_factory=list)
    total_iterations: int = 0
    total_subcalls: int = 0
