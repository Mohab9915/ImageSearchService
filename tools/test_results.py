"""Collapsing hits to products.

Each picture of a product is its own point, so a good match comes back four
or five times over -- the same dress in every colour, filling every slot the
caller asked for. These check that one product occupies one row, that the row
keeps the colour whose picture actually matched, and that the other colours
survive alongside rather than being thrown away.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagesearch import results  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        got      {got!r}")
        print(f"        expected {expected!r}")
        FAILURES.append(label)


def hit(sku, score, colour=None, **extra):
    payload = {
        "sku": sku, "product_id": int(sku.rsplit("-", 1)[-1]),
        "name": f"product {sku}", "price": 199.0, "currency": "ILS",
        "in_stock": True, "image_url": f"https://cdn.test/{sku}-{colour or 'x'}.webp",
        "ai_tenant_id": "tn_1_2",
    }
    if colour:
        payload.update({"colour": colour, "colour_in_stock": True,
                        "colour_sizes": ["38", "40"]})
    payload.update(extra)
    return {"score": score, "payload": payload}


def main() -> int:
    print("One row per product")
    rows = results.best_per_product([
        hit("CONFIGURABLE-7875", 0.91, "أسود"),
        hit("CONFIGURABLE-7875", 0.88, "بني"),
        hit("CONFIGURABLE-7875", 0.84, "أحمر"),
        hit("CONFIGURABLE-8181", 0.79, "أزرق"),
    ])
    check("two products, not four", len(rows), 2)
    check("best first", [r["sku"] for r in rows],
          ["CONFIGURABLE-7875", "CONFIGURABLE-8181"])

    print("\nThe colour that matched is the colour reported")
    check("the top hit's colour wins", rows[0]["colour"], "أسود")
    check("its picture comes with it", rows[0]["image_url"],
          "https://cdn.test/CONFIGURABLE-7875-أسود.webp")

    print("\nThe other colours are kept, not discarded")
    others = [c["colour"] for c in rows[0]["other_colours"]]
    check("both are there", sorted(others), sorted(["بني", "أحمر"]))
    check("with their own sizes",
          rows[0]["other_colours"][0]["colour_sizes"], ["38", "40"])

    print("\nOrder of arrival does not decide the winner")
    rows = results.best_per_product([
        hit("CONFIGURABLE-7875", 0.70, "بني"),
        hit("CONFIGURABLE-7875", 0.95, "أسود"),
    ])
    check("the higher score wins", rows[0]["colour"], "أسود")
    check("the loser is kept as another colour",
          [c["colour"] for c in rows[0]["other_colours"]], ["بني"])

    print("\nThe limit applies to products, not to pictures")
    rows = results.best_per_product([
        hit("A-1", 0.9, "أسود"), hit("A-1", 0.89, "بني"),
        hit("B-2", 0.8, "أسود"), hit("B-2", 0.79, "بني"),
        hit("C-3", 0.7, "أسود"),
    ], limit=2)
    check("two products come back", len(rows), 2)
    check("and they are the best two", [r["sku"] for r in rows], ["A-1", "B-2"])

    print("\nA point with no sku falls back to its product id")
    rows = results.best_per_product([
        {"score": 0.9, "payload": {"product_id": 24083, "name": "no sku"}},
    ])
    check("it is not dropped", len(rows), 1)
    check("and carries the id", rows[0]["product_id"], 24083)

    print("\nA point with neither is dropped rather than merged with others")
    rows = results.best_per_product([
        {"score": 0.9, "payload": {"name": "orphan"}},
        {"score": 0.8, "payload": {"name": "other orphan"}},
    ])
    check("nothing comes back", rows, [])

    print("\nA simple product with no colours still works")
    rows = results.best_per_product([hit("SIMPLE-1", 0.9)])
    check("one row", len(rows), 1)
    check("no colour", rows[0]["colour"], None)
    check("no other colours", rows[0]["other_colours"], [])

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
