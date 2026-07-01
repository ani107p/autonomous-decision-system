"""
Multi-provider LLM router.

Production pattern: different agents in a pipeline often use different
models -- e.g. a cheap/fast model for planning, a stronger model for
execution, and a cross-model validator (using a *different* provider than
the executor specifically to avoid correlated blind spots/hallucinations).

This router implements that pattern with three named provider slots
(openai / anthropic / gemini). Each slot returns a real LangChain chat
model if the relevant API key is present in the environment, otherwise a
deterministic offline stand-in with a distinct "reasoning profile" so the
multi-model validation behavior is still meaningfully exercised without
any network calls.
"""
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Protocol


class ChatModel(Protocol):
    name: str
    def invoke(self, prompt: str) -> str: ...


@dataclass
class CallStats:
    calls: int = 0
    cache_hits: int = 0


class _OfflineProfileLLM:
    """Deterministic offline reasoner with a configurable error rate, used
    to stand in for a real provider so the pipeline is fully runnable
    without API keys. `error_rate` simulates the base hallucination rate
    of an ungoverned single LLM call -- this is what the Validator/
    SelfCorrector agents exist to catch."""

    def __init__(self, name: str, error_rate: float = 0.0, seed: int = 0):
        self.name = name
        self.error_rate = error_rate
        self._rng = random.Random(seed)
        self.stats = CallStats()

    def invoke(self, prompt: str) -> str:
        self.stats.calls += 1
        if "[TASK=solve]" in prompt:
            return self._solve(prompt)
        if "[TASK=validate]" in prompt:
            return self._validate(prompt)
        if "[TASK=plan]" in prompt:
            return self._plan(prompt)
        return "no-op"

    def _plan(self, prompt: str) -> str:
        return "PLAN: 1) parse problem 2) compute 3) sanity-check units/magnitude"

    def _solve(self, prompt: str) -> str:
        # Extract the ground-truth answer the task harness embedded in the
        # prompt (after "GROUND_TRUTH="), then decide whether to answer
        # correctly or inject a plausible wrong answer, based on this
        # provider's configured error rate.
        marker = "GROUND_TRUTH="
        idx = prompt.find(marker)
        if idx == -1:
            return "unable to solve: no ground truth marker found"
        gt_str = prompt[idx + len(marker):].splitlines()[0].strip()
        try:
            gt = float(gt_str)
        except ValueError:
            return gt_str

        if self._rng.random() < self.error_rate:
            # Inject a plausible-but-wrong numeric answer (off by a random
            # small perturbation) -- simulates a hallucinated computation.
            perturbation = self._rng.choice([-0.15, -0.1, 0.1, 0.15, 0.2, -0.2])
            wrong = gt * (1 + perturbation) if gt != 0 else self._rng.choice([1, -1, 2])
            return f"{wrong:.2f}"
        return f"{gt:.2f}" if not float(gt).is_integer() else str(int(gt))

    def _validate(self, prompt: str) -> str:
        # Independent recomputation check: does the proposed answer match
        # the ground truth embedded in the prompt (within tolerance)?
        marker = "GROUND_TRUTH="
        answer_marker = "PROPOSED_ANSWER="
        try:
            gt = float(prompt.split(marker, 1)[1].split()[0])
            proposed = float(prompt.split(answer_marker, 1)[1].split()[0])
        except (IndexError, ValueError):
            return "INVALID: could not parse for validation"
        if abs(gt - proposed) < max(0.01, abs(gt) * 0.01):
            return "VALID"
        return f"INVALID: expected ~{gt:.2f}, got {proposed:.2f}"


def _try_external(provider: str):
    """Return a real LangChain chat model wrapper if credentials exist."""
    if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model="gpt-4o")
            return _LangChainWrapper(model, name="gpt-4o")
        except ImportError:
            return None
    if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(model="claude-sonnet-4-5")
            return _LangChainWrapper(model, name="claude-sonnet-4-5")
        except ImportError:
            return None
    if provider == "gemini" and os.environ.get("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            return _LangChainWrapper(model, name="gemini-2.0-flash")
        except ImportError:
            return None
    return None


class _LangChainWrapper:
    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.stats = CallStats()

    def invoke(self, prompt: str) -> str:
        self.stats.calls += 1
        result = self.model.invoke(prompt)
        return getattr(result, "content", str(result))


class LLMRouter:
    """
    Named model slots for a multi-agent pipeline:
      - planner:   fast/cheap model for decomposition
      - executor:  primary reasoning model
      - validator: a *different* model/provider than the executor, used to
                   cross-check the executor's output (reduces correlated
                   errors vs. a model validating itself)
    """

    def __init__(self, executor_error_rate: float = 0.22):
        self.planner: ChatModel = _try_external("openai") or _OfflineProfileLLM(
            "offline-planner", error_rate=0.0, seed=1
        )
        self.executor: ChatModel = _try_external("anthropic") or _OfflineProfileLLM(
            "offline-executor", error_rate=executor_error_rate, seed=2
        )
        # Validator is deliberately low-error -- it's doing independent
        # recomputation/verification, not open-ended generation.
        self.validator: ChatModel = _try_external("gemini") or _OfflineProfileLLM(
            "offline-validator", error_rate=0.0, seed=3
        )

    def all_stats(self) -> dict:
        return {
            "planner": getattr(self.planner, "stats", CallStats()).__dict__,
            "executor": getattr(self.executor, "stats", CallStats()).__dict__,
            "validator": getattr(self.validator, "stats", CallStats()).__dict__,
        }
