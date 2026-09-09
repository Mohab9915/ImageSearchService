from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from imagesearch import indexer, qdrant, tenants
from imagesearch.bootstrap import resolve

_seen: Dict[str, float] = {}
_seen_lock = threading.Lock()
_SEEN_TTL = 3600.0


def _duplicate(idempotency_key: Optional[str]) -> bool:
    if not idempotency_key:
        return False
    now = time.monotonic()
    with _seen_lock:
        for key, seen_at in list(_seen.items()):
            if now - seen_at > _SEEN_TTL:
                del _seen[key]
        if idempotency_key in _seen:
            return True
        _seen[idempotency_key] = now
        return False


def handle(
    ai_tenant_id: str,
    idempotency_key: Optional[str],
    action: str,
    product: Dict[str, Any],
) -> Dict[str, Any]:
    if _duplicate(idempotency_key):
        return {"status": "duplicate", "ai_tenant_id": ai_tenant_id}

    res = resolve(ai_tenant_id)
    collection = res.collection or qdrant.base_collection_name(ai_tenant_id)

    if action == "delete":
        product_id = product.get("id")
        if product_id is None:
            return {"status": "ignored", "ai_tenant_id": ai_tenant_id, "detail": "no product id"}
        indexer.delete_product(collection, product_id)
        return {"status": "deleted", "ai_tenant_id": ai_tenant_id, "collection": collection}

    if action not in ("create", "update"):
        return {"status": "ignored", "ai_tenant_id": ai_tenant_id, "detail": f"unknown action {action}"}

    tenant = tenants.lookup(ai_tenant_id)
    stats = indexer.index_products(
        collection_name=collection,
        products=[product],
        ai_tenant_id=ai_tenant_id,
        page_id=tenant.page_id if tenant else None,
        default_currency=tenant.currency if tenant else None,
        locale=tenant.locale if tenant else None,
    )
    qdrant.list_collections(refresh=True)
    return {"status": "indexed", "ai_tenant_id": ai_tenant_id, "collection": collection, **stats}
