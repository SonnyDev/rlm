"""Sandboxed Python REPL for RLM code execution."""

import io
import sys
import re
import threading
import traceback
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .types import SubCall

SAFE_BUILTINS = {
    # Types
    "str": str, "int": int, "float": float, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "bool": bool, "bytes": bytes, "bytearray": bytearray,
    "frozenset": frozenset, "complex": complex, "type": type, "object": object,
    # Iteration
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "range": range, "iter": iter, "next": next,
    # Math
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "pow": pow,
    "divmod": divmod, "len": len, "hash": hash,
    # String
    "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
    "format": format, "repr": repr, "ascii": ascii,
    # Object inspection
    "isinstance": isinstance, "issubclass": issubclass,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "delattr": delattr,
    "dir": dir, "id": id, "callable": callable, "vars": vars,
    # Exceptions
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "FileNotFoundError": FileNotFoundError, "IOError": IOError, "OSError": OSError,
    # Other
    "print": print,  # overridden below
    "super": super, "property": property, "staticmethod": staticmethod,
    "classmethod": classmethod, "any": any, "all": all,
    "__import__": __import__, "open": open,
    # Blocked
    "eval": None, "exec": None, "compile": None,
    "globals": None, "locals": None, "input": None, "breakpoint": None,
}


class LocalREPL:
    """Sandboxed Python REPL with context, llm_query, and FINAL support."""

    def __init__(self, context: str, llm_query_fn, llm_query_batched_fn):
        self._context = context
        self._llm_query_fn = llm_query_fn
        self._llm_query_batched_fn = llm_query_batched_fn
        self._subcalls: list[SubCall] = []
        self._lock = threading.Lock()
        self._final_answer: str | None = None
        self._tmpdir = tempfile.mkdtemp(prefix="rlm_repl_")

        self._namespace: dict = {}
        self._reset_namespace()

    def _reset_namespace(self):
        safe_print = self._make_safe_print()
        self._namespace = {
            "__builtins__": {**SAFE_BUILTINS, "print": safe_print},
            "context": self._context,
            "llm_query": self._llm_query,
            "llm_query_batched": self._llm_query_batched,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
        }
        # Pre-import common modules
        for mod in ["json", "re", "csv", "collections", "math", "statistics",
                     "itertools", "functools", "textwrap", "io", "os"]:
            try:
                self._namespace[mod] = __import__(mod)
            except ImportError:
                pass

    def _make_safe_print(self):
        """Create a print function that captures output."""
        return print  # Will be overridden during execute

    def _llm_query(self, prompt: str) -> str:
        result = self._llm_query_fn(prompt)
        with self._lock:
            self._subcalls.append(SubCall(prompt=prompt, response=result))
        return result

    def _llm_query_batched(self, prompts: list[str]) -> list[str]:
        results = self._llm_query_batched_fn(prompts)
        with self._lock:
            for p, r in zip(prompts, results):
                self._subcalls.append(SubCall(prompt=p, response=r))
        return results

    def _final(self, answer: str):
        self._final_answer = str(answer)

    def _final_var(self, var_name: str):
        if var_name in self._namespace:
            self._final_answer = str(self._namespace[var_name])
        else:
            raise NameError(f"Variable '{var_name}' not found in REPL namespace")

    def execute(self, code: str) -> tuple[str, list[SubCall], str | None]:
        """Execute code and return (output, subcalls, final_answer_or_none)."""
        self._subcalls = []
        self._final_answer = None

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        old_cwd = os.getcwd()

        try:
            os.chdir(self._tmpdir)
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            self._namespace["__builtins__"]["print"] = lambda *a, **kw: print(
                *a, **kw, file=stdout_capture
            )

            exec(code, self._namespace)  # noqa: S102

        except Exception:
            traceback.print_exc(file=stderr_capture)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.chdir(old_cwd)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        if errors:
            output += f"\n[STDERR]\n{errors}"

        return output, list(self._subcalls), self._final_answer

    def cleanup(self):
        import shutil
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass


def extract_code_blocks(text: str) -> list[str]:
    """Extract ```repl or ```python code blocks from LLM response."""
    pattern = r"```(?:repl|python)\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return blocks


def extract_reasoning(text: str) -> str:
    """Extract text before the first code block as reasoning."""
    match = re.search(r"```(?:repl|python)", text)
    if match:
        return text[:match.start()].strip()
    return text.strip()
