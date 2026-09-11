"""In-memory request-rate limiter.

Fixed window, per identifier (bearer token if present, else client IP).

ponytail: counters live in one process's memory, so with WEB_CONCURRENCY > 1
each gunicorn worker enforces its own limit -- effective throughput is
limit * workers, not one shared cap. Move to a shared store (Redis) if a hard
global limit is ever required.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple


class RateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._lock = threading.Lock()
        self._buckets: Dict[str, Tuple[int, int]] = {}

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        """True if this request is under the limit. Always records the hit."""
        if self.limit <= 0:
            return True
        window = int((now if now is not None else time.time()) // 60)
        with self._lock:
            win, count = self._buckets.get(key, (window, 0))
            if win != window:
                win, count = window, 0
            count += 1
            self._buckets[key] = (win, count)
            return count <= self.limit
