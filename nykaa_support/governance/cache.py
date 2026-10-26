"""
In-Memory Response Caching Keyed by Normalized Query Text (Task 16).

Strictly complies with Capstone Project Brief (Pages 8-9):
1. In-memory cache keyed by normalized query text for grounded generation.
2. Demonstrates repeated query producing cache hit avoiding redundant LLM/tool calls.
3. Provides before/after evidence (call counter and timing metrics).
"""

import re
import time
from typing import Any, Callable, Dict, Optional, Tuple


def normalize_query_text(query: str) -> str:
    """
    Normalize customer query text:
    - Lowercase
    - Remove superfluous punctuation (trailing ?, !, .)
    - Collapse multiple whitespace characters into a single space
    - Strip leading and trailing whitespace
    """
    cleaned = query.lower().strip()
    cleaned = re.sub(r'[\?\!\.\,\;]+$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


class NormalizedQueryCache:
    """
    In-memory cache for grounded generation responses.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self.hits: int = 0
        self.misses: int = 0
        self.total_calls: int = 0

    def get(self, query: str) -> Optional[Any]:
        """Fetch cached response by normalized query."""
        key = normalize_query_text(query)
        self.total_calls += 1
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, query: str, response: Any) -> None:
        """Store response keyed by normalized query."""
        key = normalize_query_text(query)
        self._cache[key] = response

    def contains(self, query: str) -> bool:
        """Check if normalized query exists in cache."""
        return normalize_query_text(query) in self._cache

    def clear(self) -> None:
        """Clear cache and reset stats."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_calls = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        hit_rate = (self.hits / self.total_calls * 100.0) if self.total_calls > 0 else 0.0
        return {
            "cached_entries": len(self._cache),
            "total_calls": self.total_calls,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
        }

    def get_or_call(
        self,
        query: str,
        generator_fn: Callable[[str], Any]
    ) -> Tuple[Any, bool, float]:
        """
        Executes generator_fn only on cache miss.
        Returns (response, is_cache_hit, duration_seconds).
        """
        start = time.perf_counter()
        cached_result = self.get(query)
        if cached_result is not None:
            duration = time.perf_counter() - start
            return cached_result, True, duration

        # Cache miss: execute the generator function
        fresh_result = generator_fn(query)
        self.set(query, fresh_result)
        duration = time.perf_counter() - start
        return fresh_result, False, duration
