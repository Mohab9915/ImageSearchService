"""Where a tenant's catalog credentials come from.

Read-only against the chatbot's Postgres. This service never writes there --
the tenant row, the page row and the collection name recorded on it all
belong to the chatbot, and a bootstrap that repointed a page would take a
live catalog away from it.

If DATABASE_URL is unset the service still works; the caller then has to send
`laravel_base_url` and `laravel_token` with a bootstrap request.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from imagesearch.config import DATABASE_URL

logger = logging.getLogger(__name__)

_engine = None


@dataclass
class Tenant:
    ai_tenant_id: str
    laravel_base_url: str
    laravel_token: str
    page_id: Optional[str] = None
    currency: Optional[str] = None
    locale: Optional[str] = None
    is_active: bool = True


def _get_engine():
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            return None
        from sqlalchemy import create_engine

        # pre_ping because Railway's proxy drops idle connections, and this
        # service is idle between bootstraps by design.
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=2,
                                max_overflow=2, pool_recycle=300)
    return _engine


def lookup(ai_tenant_id: str) -> Optional[Tenant]:
    """The tenant row, or None if there is no database or no such tenant."""
    engine = _get_engine()
    if engine is None or not ai_tenant_id:
        return None

    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT ai_tenant_id, laravel_api_base_url, laravel_api_token, "
                     "       facebook_page_id, currency, default_locale, is_active "
                     "FROM tenants WHERE ai_tenant_id = :t"),
                {"t": str(ai_tenant_id)},
            ).fetchone()
    except Exception as e:
        logger.error("[TENANTS] Lookup of %s failed: %s", ai_tenant_id, e)
        return None

    if not row:
        return None

    return Tenant(
        ai_tenant_id=row[0],
        laravel_base_url=row[1] or "",
        laravel_token=row[2] or "",
        page_id=str(row[3]) if row[3] else None,
        currency=row[4],
        locale=row[5],
        is_active=bool(row[6]),
    )


def collection_on_page(page_id: str) -> Optional[str]:
    """The collection the chatbot's page row currently points at.

    Authoritative when it is set: a rebuild repoints the page, and this names
    the collection that finished rather than one that was abandoned midway.
    """
    engine = _get_engine()
    if engine is None or not page_id:
        return None

    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text('SELECT qdrant_collection_name FROM "PagesTable" WHERE page_id = :p'),
                {"p": str(page_id)},
            ).fetchone()
        return str(row[0]) if row and row[0] else None
    except Exception as e:
        logger.warning("[TENANTS] Could not read collection for page %s: %s", page_id, e)
        return None
