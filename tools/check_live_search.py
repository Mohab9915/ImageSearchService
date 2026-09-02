"""End-to-end search against the real Qdrant, without loading SigLIP.

The trick: take a vector that is already stored in a tenant's collection and
use it as the query. That is exactly what the model would produce for that
product's own picture, so it exercises the whole path -- routing, tenant
resolution, the Qdrant search, the collapse to one row per product -- and the
only thing stubbed is the forward pass.

Read-only, and it never downloads the model, so it is safe to run anywhere.

    python tools/check_live_search.py [tenant_id]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from imagesearch import bootstrap, embedding, qdrant  # noqa: E402
from imagesearch.config import QDRANT_URL  # noqa: E402


def a_stored_vector(collection):
    """One point's vector and payload, straight out of the collection."""
    resp = httpx.post(
        f"{QDRANT_URL}/collections/{collection}/points/scroll",
        json={"limit": 1, "with_payload": True, "with_vector": True},
        headers=qdrant.headers(), timeout=30.0,
    )
    resp.raise_for_status()
    points = resp.json()["result"]["points"]
    if not points:
        return None, None
    return points[0].get("vector"), points[0].get("payload") or {}


def main() -> int:
    from fastapi.testclient import TestClient

    tenant = sys.argv[1] if len(sys.argv) > 1 else "tn_24_30"

    res = bootstrap.resolve(tenant, refresh=True)
    print(f"tenant     {tenant}")
    print(f"collection {res.collection}  ({res.points} points, indexed={res.indexed})")
    if not res.indexed:
        print("\nThis tenant has nothing indexed -- pick another, or bootstrap it.")
        return 1

    vector, payload = a_stored_vector(res.collection)
    if not vector:
        print("Could not read a vector back out of the collection.")
        return 1

    print(f"query      the stored vector for {payload.get('sku')} "
          f"({payload.get('colour') or 'no colour'})")
    print(f"           \"{(payload.get('name') or '')[:60]}\"")

    # Stand in for the model. Everything downstream is the real thing.
    import main as service
    embedding.embed_image_url = lambda url, timeout=15.0: vector

    client = TestClient(service.app)

    print("\n--- GET /health")
    r = client.get("/health")
    print(f"    {r.status_code} {json.dumps(r.json(), ensure_ascii=False)}")

    print(f"\n--- GET /tenants/{tenant}")
    r = client.get(f"/tenants/{tenant}")
    body = r.json()
    print(f"    {r.status_code} collection={body.get('collection')} "
          f"points={body.get('points')} indexed={body.get('indexed')} "
          f"source={body.get('source')}")

    print(f"\n--- POST /tenants/{tenant}/bootstrap  (must not embed)")
    r = client.post(f"/tenants/{tenant}/bootstrap", json={})
    body = r.json()
    print(f"    {r.status_code} status={body.get('status')} -- {body.get('detail')}")
    if body.get("status") != "skipped":
        print("    !! expected 'skipped' for an already-indexed tenant")
        return 1

    print("\n--- POST /search")
    r = client.post("/search", json={
        "ai_tenant_id": tenant,
        "image_url": "https://example.test/ignored-the-embedder-is-stubbed.jpg",
        "limit": 5,
    })
    if r.status_code != 200:
        print(f"    {r.status_code} {r.text[:400]}")
        return 1

    body = r.json()
    print(f"    200  {body['count']} product(s) from {body['collection']}\n")
    for row in body["results"]:
        colours = ", ".join(c["colour"] for c in row["other_colours"]) or "-"
        print(f"      {row['score']:.4f}  {row['sku']:<22} {(row['title'] or '')[:34]:<34}"
              f"  colour={row['colour'] or '-'}  others={colours}")

    top = body["results"][0]
    if top["sku"] != payload.get("sku"):
        print(f"\n    !! the product's own vector did not return it first "
              f"(got {top['sku']}, wanted {payload.get('sku')})")
        return 1

    print(f"\n    the query vector's own product came back first, at {top['score']}")
    print("\nNothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
