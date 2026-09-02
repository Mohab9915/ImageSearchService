"""Talking to the shared Qdrant.

This service does not own a vector store. It reads and writes the same one
the chatbot uses, under the same collection naming convention, so a tenant
indexed by either is searchable by both.

The convention: a tenant's products live in `tenant_products_{ai_tenant_id}`.
A rebuild builds into `tenant_products_{ai_tenant_id}_{unix_timestamp}` and
repoints the pages at it once it completes, so the newest stamped name -- not
the bare one -- is a tenant's current collection.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from imagesearch.config import (
    LOCK_COLLECTION, QDRANT_API_KEY, QDRANT_URL, VECTOR_SIZE,
)

logger = logging.getLogger(__name__)


def headers() -> dict:
    h = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        h["api-key"] = QDRANT_API_KEY
    return h


def base_collection_name(ai_tenant_id: str) -> str:
    """The unstamped name for a tenant.

    Sanitised the same way the chatbot sanitises it -- a tenant id that
    round-trips differently here would address a collection that does not
    exist.
    """
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", ai_tenant_id or "").strip("_").lower()
    return f"tenant_products_{safe or 'unknown'}"


_names: List[str] = []
_names_at = 0.0


def list_collections(ttl: int = 30, refresh: bool = False) -> List[str]:
    """Every collection Qdrant currently holds."""
    global _names, _names_at

    now = time.time()
    if _names and not refresh and now - _names_at < ttl:
        return _names
    try:
        resp = httpx.get(f"{QDRANT_URL}/collections", headers=headers(), timeout=10.0)
        resp.raise_for_status()
        names = [c["name"] for c in resp.json()["result"]["collections"]]
    except Exception as e:
        logger.warning("[QDRANT] Could not list collections: %s", e)
        return _names

    _names, _names_at = names, now
    return names


def collection_for_tenant(ai_tenant_id: str, refresh: bool = False) -> Optional[str]:
    """A tenant's current collection, newest rebuild first, or None.

    Returning None is the signal the whole service turns on: it means this
    tenant has never been indexed, and only then may we embed anything.
    """
    if not ai_tenant_id:
        return None

    prefix = base_collection_name(ai_tenant_id)
    stamped: List[tuple] = []
    plain = None

    for name in list_collections(refresh=refresh):
        if name == prefix:
            plain = name
        elif name.startswith(prefix + "_"):
            suffix = name[len(prefix) + 1:]
            if suffix.isdigit():
                stamped.append((int(suffix), name))

    if stamped:
        return max(stamped)[1]
    return plain


def collection_info(collection_name: str) -> Optional[Dict[str, Any]]:
    """Qdrant's own view of a collection, or None if it is not there."""
    try:
        resp = httpx.get(
            f"{QDRANT_URL}/collections/{collection_name}",
            headers=headers(), timeout=10.0,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("result")
    except Exception as e:
        logger.warning("[QDRANT] Could not read %s: %s", collection_name, e)
        return None


def point_count(collection_name: str) -> int:
    info = collection_info(collection_name)
    return int((info or {}).get("points_count") or 0)


def ensure_collection(collection_name: str, size: int = VECTOR_SIZE) -> None:
    """Create the collection if it is not already there.

    Same vector parameters as the chatbot's: any difference in size or
    distance metric and the two services could not share the collection.
    """
    if collection_info(collection_name) is not None:
        return

    resp = httpx.put(
        f"{QDRANT_URL}/collections/{collection_name}",
        json={"vectors": {"size": size, "distance": "Cosine"}},
        headers=headers(), timeout=20.0,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Could not create collection {collection_name}: "
            f"{resp.status_code} {resp.text[:300]}"
        )
    logger.info("[QDRANT] Created collection %s", collection_name)
    list_collections(refresh=True)


def upsert(collection_name: str, points: List[dict]) -> None:
    if not points:
        return
    resp = httpx.put(
        f"{QDRANT_URL}/collections/{collection_name}/points?wait=true",
        json={"points": points}, headers=headers(), timeout=60.0,
    )
    if resp.status_code not in (200, 202):
        raise RuntimeError(
            f"Upsert into {collection_name} failed: "
            f"{resp.status_code} {resp.text[:300]}"
        )


def search(
    collection_name: str,
    vector: List[float],
    limit: int,
    score_threshold: float,
) -> List[Dict[str, Any]]:
    """Nearest products, as Qdrant returns them.

    Payloads are read, not reshaped, here -- see `results.py` for the shape
    this service hands back.
    """
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                f"{QDRANT_URL}/collections/{collection_name}/points/search",
                json={
                    "vector": vector,
                    "limit": limit,
                    "with_payload": True,
                    "score_threshold": score_threshold,
                },
                headers=headers(),
            )
        if resp.status_code != 200:
            logger.warning("[QDRANT] Search on %s failed: %s %s",
                           collection_name, resp.status_code, resp.text[:200])
            return []
        return resp.json().get("result", [])
    except Exception as e:
        logger.error("[QDRANT] Search on %s errored: %s", collection_name, e)
        return []


# --------------------------------------------------------------- locking
# A bootstrap can run for tens of minutes. Two requests arriving in that
# window -- or two instances of this service behind one Railway load balancer
# -- would otherwise both decide the tenant is unindexed and both start
# embedding the same catalog.
#
# The lock lives in its own one-dimensional collection rather than as a
# marker point inside the tenant's own, so it can never come back as a search
# result. Qdrant has no lock primitive; this is a lease with an expiry, which
# is enough because the cost of a rare double-index is wasted CPU, not
# corruption -- points are upserted under deterministic ids.

def _lock_id(ai_tenant_id: str) -> str:
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"bootstrap-lock:{ai_tenant_id}"))


def read_lock(ai_tenant_id: str) -> Optional[Dict[str, Any]]:
    """The lease on this tenant, if one is held."""
    try:
        resp = httpx.get(
            f"{QDRANT_URL}/collections/{LOCK_COLLECTION}/points/{_lock_id(ai_tenant_id)}",
            headers=headers(), timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        return (resp.json().get("result") or {}).get("payload")
    except Exception:
        return None


def take_lock(ai_tenant_id: str, ttl: int) -> bool:
    """Claim the right to bootstrap this tenant. False if someone else holds it."""
    held = read_lock(ai_tenant_id)
    if held and held.get("state") == "running":
        age = time.time() - float(held.get("started_at") or 0)
        if age < ttl:
            return False
        logger.warning("[BOOTSTRAP] Stale lock on %s (%.0fs old) -- taking it over",
                       ai_tenant_id, age)

    ensure_collection(LOCK_COLLECTION, size=1)
    upsert(LOCK_COLLECTION, [{
        "id": _lock_id(ai_tenant_id),
        "vector": [0.0],
        "payload": {
            "ai_tenant_id": ai_tenant_id,
            "state": "running",
            "started_at": time.time(),
        },
    }])
    return True


def release_lock(ai_tenant_id: str, state: str, detail: Optional[str] = None) -> None:
    """Record how the bootstrap ended. Kept rather than deleted: a failed
    bootstrap is the thing you most want to be able to read afterwards."""
    try:
        upsert(LOCK_COLLECTION, [{
            "id": _lock_id(ai_tenant_id),
            "vector": [0.0],
            "payload": {
                "ai_tenant_id": ai_tenant_id,
                "state": state,
                "detail": detail,
                "finished_at": time.time(),
            },
        }])
    except Exception as e:
        logger.warning("[BOOTSTRAP] Could not release lock for %s: %s", ai_tenant_id, e)
