"""Deciding whether a tenant needs indexing, and doing it if so.

The rule this service is built around: **embed a tenant's products if and
only if that tenant has no collection in Qdrant.** A tenant the chatbot has
already indexed is left alone -- its vectors are already there, re-embedding
them would cost hours of CPU to produce the same numbers, and doing it while
the chatbot is reading would serve a half-built catalog to customers.

`resolve` answers "which collection is this tenant's, and is it usable?".
`bootstrap_if_missing` acts on that answer.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from imagesearch import qdrant, tenants
from imagesearch.config import ALLOW_FORCE_REINDEX, BOOTSTRAP_LOCK_TTL

logger = logging.getLogger(__name__)

# Guards the decide-then-act window inside one process. The Qdrant lease in
# qdrant.py guards it across processes; this just avoids two threads of the
# same worker racing to take that lease.
_local = threading.Lock()

# Bootstraps in flight, by tenant, so a status call can report on one.
_running: Dict[str, Dict[str, Any]] = {}


@dataclass
class Resolution:
    """What we know about a tenant's index."""
    ai_tenant_id: str
    collection: Optional[str]      # the name to search, if there is one
    exists: bool                   # Qdrant has it
    points: int                    # how many vectors are in it
    indexed: bool                  # exists AND has points -- safe to search
    source: str                    # how the name was arrived at


def resolve(ai_tenant_id: str, refresh: bool = False) -> Resolution:
    """Find a tenant's collection without creating anything.

    Two sources, and they can disagree. The page row in Postgres is what the
    chatbot itself reads through, so it wins; the Qdrant names are the
    fallback for a tenant whose page row was never written -- and the only
    source at all when this service runs without a database.
    """
    name: Optional[str] = None
    source = "none"

    tenant = tenants.lookup(ai_tenant_id)
    if tenant and tenant.page_id:
        recorded = tenants.collection_on_page(tenant.page_id)
        if recorded:
            name, source = recorded, "page_row"

    if not name:
        derived = qdrant.collection_for_tenant(ai_tenant_id, refresh=refresh)
        if derived:
            name, source = derived, "qdrant_name"

    if not name:
        return Resolution(ai_tenant_id, None, False, 0, False, "none")

    info = qdrant.collection_info(name)
    if info is None:
        # The page names a collection that is no longer there. Fall back to
        # whatever Qdrant actually holds before concluding the tenant is
        # unindexed -- a rebuild may have moved it.
        derived = qdrant.collection_for_tenant(ai_tenant_id, refresh=True)
        if derived and derived != name:
            info = qdrant.collection_info(derived)
            if info is not None:
                name, source = derived, "qdrant_name_after_stale_page_row"

    if info is None:
        return Resolution(ai_tenant_id, name, False, 0, False, source + "_missing")

    points = int(info.get("points_count") or 0)
    return Resolution(ai_tenant_id, name, True, points, points > 0, source)


def status(ai_tenant_id: str) -> Dict[str, Any]:
    """Everything a caller might want to know before searching."""
    res = resolve(ai_tenant_id)
    out = asdict(res)
    out["running"] = _running.get(ai_tenant_id)
    lock = qdrant.read_lock(ai_tenant_id)
    out["last_bootstrap"] = lock
    return out


def _client_for(ai_tenant_id: str,
                laravel_base_url: Optional[str],
                laravel_token: Optional[str]):
    """A catalog client from the request, or from the tenants table."""
    from imagesearch.catalog import CatalogClient

    if laravel_base_url and laravel_token:
        return CatalogClient(laravel_base_url, laravel_token, ai_tenant_id), None, None

    tenant = tenants.lookup(ai_tenant_id)
    if not tenant:
        raise ValueError(
            f"Unknown tenant {ai_tenant_id}. Either set DATABASE_URL so this "
            f"service can read the tenants table, or send laravel_base_url "
            f"and laravel_token with the request."
        )
    if not tenant.laravel_base_url or not tenant.laravel_token:
        raise ValueError(f"Tenant {ai_tenant_id} has no Laravel credentials on file")

    return (CatalogClient(tenant.laravel_base_url, tenant.laravel_token, ai_tenant_id),
            tenant.currency, tenant.locale)


def bootstrap_if_missing(
    ai_tenant_id: str,
    *,
    laravel_base_url: Optional[str] = None,
    laravel_token: Optional[str] = None,
    limit: int = 0,
    force: bool = False,
) -> Dict[str, Any]:
    """Index a tenant's catalog, but only if it has never been indexed.

    Returns a dict whose `status` is one of:
      skipped   -- already indexed; nothing was embedded
      locked    -- another process is bootstrapping this tenant right now
      completed -- the catalog was pulled and embedded
      failed    -- something went wrong; `detail` says what

    `force` re-embeds a tenant that already has a collection. It is refused
    unless ALLOW_FORCE_REINDEX is set on the service, because the ordinary
    way to rebuild a catalog is the chatbot's own sync, which builds into a
    fresh collection and swaps only on success. This one writes in place.
    """
    if force and not ALLOW_FORCE_REINDEX:
        return {
            "status": "failed",
            "ai_tenant_id": ai_tenant_id,
            "detail": "force re-index is disabled; set ALLOW_FORCE_REINDEX=true "
                      "to allow it, and only when nothing is reading the catalog",
        }

    with _local:
        res = resolve(ai_tenant_id, refresh=True)

        if res.indexed and not force:
            logger.info("[BOOTSTRAP] %s already indexed (%s, %d points) -- skipping",
                        ai_tenant_id, res.collection, res.points)
            return {
                "status": "skipped",
                "ai_tenant_id": ai_tenant_id,
                "collection": res.collection,
                "points": res.points,
                "detail": "tenant already has a collection; nothing was embedded",
            }

        if not qdrant.take_lock(ai_tenant_id, BOOTSTRAP_LOCK_TTL):
            return {
                "status": "locked",
                "ai_tenant_id": ai_tenant_id,
                "detail": "another bootstrap for this tenant is already running",
            }

        _running[ai_tenant_id] = {"phase": "starting", "indexed": 0}

    # An existing-but-empty collection is reused rather than renamed: the
    # chatbot's page row may already point at it.
    collection = res.collection or qdrant.base_collection_name(ai_tenant_id)

    try:
        client, currency, locale = _client_for(ai_tenant_id, laravel_base_url, laravel_token)

        # Fail before creating anything if the credentials are wrong. A
        # collection created for a tenant we cannot pull would make every
        # later call believe the tenant is indexed.
        if not client.reachable():
            raise RuntimeError(
                f"Laravel catalog at {client.base_url} did not answer -- "
                f"check the base url and token"
            )

        from imagesearch.catalog import pull_products
        from imagesearch.indexer import index_products

        _running[ai_tenant_id] = {"phase": "indexing", "collection": collection}
        logger.info("[BOOTSTRAP] %s -> %s : pulling and embedding",
                    ai_tenant_id, collection)

        tenant = tenants.lookup(ai_tenant_id)
        stats = index_products(
            collection_name=collection,
            products=pull_products(client),
            ai_tenant_id=ai_tenant_id,
            page_id=tenant.page_id if tenant else None,
            limit=limit,
            force_reindex=force,
            default_currency=currency,
            locale=locale,
        )

        qdrant.list_collections(refresh=True)
        qdrant.release_lock(ai_tenant_id, "completed")
        logger.info("[BOOTSTRAP] %s finished: %s", ai_tenant_id, stats)

        return {
            "status": "completed",
            "ai_tenant_id": ai_tenant_id,
            "collection": collection,
            "points": qdrant.point_count(collection),
            **stats,
        }

    except Exception as e:
        logger.error("[BOOTSTRAP] %s failed: %s", ai_tenant_id, e)
        qdrant.release_lock(ai_tenant_id, "failed", str(e)[:500])
        return {
            "status": "failed",
            "ai_tenant_id": ai_tenant_id,
            "collection": collection,
            "detail": str(e)[:500],
        }
    finally:
        _running.pop(ai_tenant_id, None)
