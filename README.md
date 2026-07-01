"""
Response caching for LLM calls.

Keys on (model_name, prompt) so repeated/near-duplicate task runs (common
in decision pipelines re-evaluating similar scenarios, or re-running a
benchmark) skip the model call entirely. Backed by `diskcache` so it
persists across process runs, same as a real deployment would want for
cost control.
"""
from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import diskcache

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / ".llm_cache"


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    total_saved_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class CachingLLM:
    """Wraps any object with `.invoke(prompt) -> str` and adds a disk cache."""

    def __init__(self, inner, simulated_call_latency: float = 0.35):
        self.inner = inner
        self.name = getattr(inner, "name", "unknown")
        self.simulated_call_latency = simulated_call_latency
        self._cache = diskcache.Cache(str(CACHE_DIR))
        self.metrics = CacheMetrics()

    @staticmethod
    def _key(model_name: str, prompt: str) -> str:
        h = hashlib.sha256(f"{model_name}::{prompt}".encode()).hexdigest()
        return f"llmcache:{h}"

    def invoke(self, prompt: str) -> str:
        key = self._key(self.name, prompt)
        cached = self._cache.get(key)
        if cached is not None:
            self.metrics.hits += 1
            self.metrics.total_saved_seconds += self.simulated_call_latency
            return cached

        self.metrics.misses += 1
        # Simulate realistic network/inference latency for offline models so
        # the async-batching benchmark produces a meaningful wall-clock
        # comparison (real API calls already have this latency for free).
        if not hasattr(self.inner, "model"):  # offline/local models only
            time.sleep(self.simulated_call_latency)
        result = self.inner.invoke(prompt)
        self._cache.set(key, result, expire=3600)
        return result

    def clear(self):
        self._cache.clear()
