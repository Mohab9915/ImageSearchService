"""Pulling a tenant's catalog out of their Laravel backend.

Only used by a bootstrap. Once a tenant has a collection, the chatbot's own
catalog sync keeps it current and this code never runs for them again.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import httpx

logger = logging.getLogger(__name__)

# A catalog page is far heavier than any other call to these backends: one
# page of products with many variants and images was measured at 31s on a
# live tenant, with its neighbours at 2-4s. At an ordinary timeout that page
# fails every retry, and a page that cannot be fetched aborts the whole pull.
CATALOG_TIMEOUT = float(os.environ.get("LARAVEL_CATALOG_TIMEOUT", "60"))

# Ask for fewer products per request each time a page fails. A backend that
# cannot assemble 100 products in one response -- a hard cap on `limit`, a
# memory ceiling, a gateway timeout -- fails an identical retry identically,
# so retrying unchanged only burns the budget.
PAGE_SIZE_LADDER = [100, 50, 25, 10]
MAX_RETRIES = int(os.environ.get("PULL_MAX_RETRIES", "4"))
MAX_PAGES = int(os.environ.get("PULL_MAX_PAGES", "2000"))


class CatalogClient:
    """The products half of the chatbot's LaravelClient, and nothing else."""

    def __init__(self, base_url: str, api_token: str, ai_tenant_id: str = ""):
        self.base_url = (base_url or "").rstrip("/")
        self.api_token = api_token
        self.ai_tenant_id = ai_tenant_id

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }

    def list_products_page(self, *, limit: int, page: int) -> dict:
        resp = httpx.get(
            f"{self.base_url}/products",
            params={"limit": limit, "page": page},
            headers=self._headers(),
            timeout=CATALOG_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def reachable(self) -> bool:
        """One cheap call, so a bootstrap fails fast on bad credentials
        rather than after taking the lock and creating a collection."""
        try:
            self.list_products_page(limit=1, page=1)
            return True
        except Exception as e:
            logger.warning("[CATALOG] %s unreachable: %s", self.base_url, e)
            return False


def _items(data: Any) -> List[Dict[str, Any]]:
    items = data.get("data", data) if isinstance(data, dict) else data
    return items if isinstance(items, list) else []


def pull_products(client: CatalogClient) -> Iterator[Dict[str, Any]]:
    """Stream every product, page by page.

    Sequential rather than concurrent: a bootstrap is a one-off, and the
    embedding that follows costs orders of magnitude more than the fetch, so
    there is nothing to win here and a slower pull is gentler on a merchant's
    backend.
    """
    ladder = list(PAGE_SIZE_LADDER)
    size_idx = 0
    page = 1
    total = 0

    while page <= MAX_PAGES:
        data = None
        attempt = 0

        while attempt < MAX_RETRIES:
            try:
                data = client.list_products_page(limit=ladder[size_idx], page=page)
                break
            except Exception as e:
                # Only resize while still on the first page: `page` indexes
                # into fixed-size pages, so changing the size later would
                # silently skip or repeat products.
                if page == 1 and size_idx + 1 < len(ladder):
                    size_idx += 1
                    logger.warning("[CATALOG] page 1 failed (%s) -- retrying at %d per page",
                                   e, ladder[size_idx])
                    continue

                attempt += 1
                if attempt >= MAX_RETRIES:
                    raise RuntimeError(
                        f"Gave up pulling page {page} after {MAX_RETRIES} attempts: {e}"
                    ) from e
                wait_for = min(2 ** attempt, 30)
                logger.warning("[CATALOG] page %d failed (attempt %d/%d): %s -- retrying in %ss",
                               page, attempt, MAX_RETRIES, e, wait_for)
                time.sleep(wait_for)

        items = _items(data)
        if not items:
            break

        for product in items:
            total += 1
            yield product

        logger.info("[CATALOG] pulled page %d (%d products so far)", page, total)
        if len(items) < ladder[size_idx]:
            break
        page += 1

    logger.info("[CATALOG] pull finished: %d products", total)
