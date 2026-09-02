"""What this service would do, right now, against the real Qdrant.

Read-only. It resolves every tenant it can see and prints the decision a
bootstrap would make, without embedding anything. Run it before pointing the
service at production to confirm it reuses what is there rather than building
beside it.

    python tools/check_live.py [tenant_id ...]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagesearch import bootstrap, qdrant  # noqa: E402
from imagesearch.config import DATABASE_URL, QDRANT_URL  # noqa: E402


def tenants_in_qdrant():
    """Tenant ids inferred from the collection names that exist."""
    found = set()
    for name in qdrant.list_collections(refresh=True):
        if not name.startswith("tenant_products_"):
            continue
        rest = name[len("tenant_products_"):]
        parts = rest.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) >= 9:
            rest = parts[0]          # strip a rebuild timestamp
        found.add(rest)
    return sorted(found)


def main() -> int:
    print(f"qdrant   {QDRANT_URL}")
    print(f"database {'set' if DATABASE_URL else 'not set (falling back to collection names)'}")

    names = qdrant.list_collections(refresh=True)
    if not names:
        print("\nCould not reach Qdrant.")
        return 1

    print(f"\n{len(names)} collection(s):")
    for name in sorted(names):
        print(f"  {name:<44} {qdrant.point_count(name):>7} points")

    wanted = sys.argv[1:] or tenants_in_qdrant()
    print(f"\nresolving {len(wanted)} tenant(s):\n")

    header = f"  {'tenant':<14} {'collection':<44} {'points':>7}  decision"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for tenant_id in wanted:
        res = bootstrap.resolve(tenant_id)
        decision = ("SKIP  (already indexed)" if res.indexed
                    else "EMBED (nothing indexed)")
        print(f"  {tenant_id:<14} {str(res.collection or '-'):<44} "
              f"{res.points:>7}  {decision}")

    print("\nNothing was written. A tenant marked SKIP is never re-embedded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
