"""Small in-process rate limiter. No extra dependency."""
from __future__ import annotations

import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, *keys: str) -> bool:
        now = time.monotonic()
        fresh: dict[str, list[float]] = {}
        for key in keys:
            kept = [t for t in self._hits[key] if now - t < self.window_s]
            if len(kept) >= self.limit:
                self._hits[key] = kept
                return False
            fresh[key] = kept
        for key, kept in fresh.items():
            kept.append(now)
            self._hits[key] = kept
        return True
