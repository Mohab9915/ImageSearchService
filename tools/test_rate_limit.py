"""The fixed-window rate limiter.

Checks the counting and window-reset logic directly, without spinning up the
whole FastAPI app (which needs the model loaded).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagesearch.ratelimit import RateLimiter  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        got      {got!r}")
        print(f"        expected {expected!r}")
        FAILURES.append(label)


def main() -> int:
    print("Under the limit, every request is allowed")
    rl = RateLimiter(3)
    check("1st", rl.allow("a", now=0), True)
    check("2nd", rl.allow("a", now=0), True)
    check("3rd", rl.allow("a", now=0), True)

    print("\nOver the limit, further requests in the same window are refused")
    check("4th is refused", rl.allow("a", now=0), False)
    check("5th is still refused", rl.allow("a", now=10), False)

    print("\nA new window resets the count")
    check("next minute is allowed again", rl.allow("a", now=60), True)

    print("\nEach key has its own bucket")
    check("a different key is unaffected by a's limit",
          rl.allow("b", now=0), True)

    print("\nlimit=0 disables the limiter")
    unlimited = RateLimiter(0)
    check("always allowed", all(unlimited.allow("x", now=0) for _ in range(100)), True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
