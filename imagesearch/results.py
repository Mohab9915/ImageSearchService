"""Turning Qdrant hits into an answer a caller can use.

The payload the chatbot writes carries the whole raw product, which is far
too much to hand back on every hit. What comes out here is the part a
merchant integration actually needs, plus the colour fields -- because each
picture of a configurable product is indexed separately, so *which* picture
matched is the difference between "we have this dress" and "we have this
dress in the colour you photographed".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def to_row(match: Dict[str, Any]) -> Dict[str, Any]:
    p = match.get("payload") or {}
    return {
        "sku": p.get("sku") or p.get("product_sku") or str(p.get("product_id") or ""),
        "product_id": p.get("product_id"),
        "title": p.get("name") or p.get("title") or "",
        "price": p.get("price"),
        "currency": p.get("currency"),
        "in_stock": p.get("in_stock"),
        "qty": p.get("qty"),
        "type": p.get("type"),
        "image_url": p.get("image_url") or "",
        "url": p.get("url"),
        "ai_tenant_id": p.get("ai_tenant_id"),
        "score": round(float(match.get("score") or 0), 4),
        # Which colour's picture this was, when the product has colours.
        "colour": p.get("colour"),
        "colour_swatch": p.get("colour_swatch"),
        "colour_in_stock": p.get("colour_in_stock"),
        "colour_sizes": p.get("colour_sizes") or [],
        "variant_id": p.get("variant_id"),
    }


def best_per_product(
    matches: List[Dict[str, Any]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """One row per product, keeping the colour that actually matched.

    A product is indexed once per picture, so a good match comes back several
    times over -- the same dress in four colours filling every slot. Collapse
    to the product, keeping its best-scoring picture, and record the other
    colours alongside so a caller can still see them.
    """
    best: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for match in matches:
        row = to_row(match)
        key = row["sku"] or f"id:{row['product_id']}"
        if not key or key == "id:None":
            continue

        if key not in best:
            best[key] = row
            best[key]["other_colours"] = []
            order.append(key)
            continue

        kept = best[key]
        loser = row
        if row["score"] > kept["score"]:
            row["other_colours"] = kept["other_colours"]
            best[key] = row
            loser = kept

        if loser.get("colour") and loser["colour"] != best[key].get("colour"):
            best[key]["other_colours"].append({
                "colour": loser["colour"],
                "colour_in_stock": loser.get("colour_in_stock"),
                "colour_sizes": loser.get("colour_sizes") or [],
                "image_url": loser.get("image_url"),
                "score": loser["score"],
            })

    rows = sorted((best[k] for k in order), key=lambda r: -r["score"])
    return rows[:limit] if limit else rows


def split_exact_similar(rows, threshold, limit=None):
    exact = [r for r in rows if r["score"] >= threshold]
    exact_skus = {r["sku"] for r in exact}
    similar = [r for r in rows if r["score"] < threshold and r["sku"] not in exact_skus]
    return exact[:limit], similar[:limit]
