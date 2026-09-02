"""The rule the service exists to enforce: embed only when nothing is indexed.

Every path through `bootstrap_if_missing` is exercised here with Qdrant, the
tenants table and the indexer all stubbed, so the test is about the decision
and not about the network. If any of these start passing an indexer call
through for a tenant that already has points, a deploy of this service would
re-embed live catalogs.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagesearch import bootstrap, qdrant, tenants  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        got      {got!r}")
        print(f"        expected {expected!r}")
        FAILURES.append(label)


class Stub:
    """Stands in for Qdrant, the tenants table and the indexer at once."""

    def __init__(self, *, collections=(), points=None, page_collection=None,
                 tenant=None, lock=None):
        self.collections = list(collections)
        self.points = dict(points or {})
        self.page_collection = page_collection
        self.tenant = tenant
        self.lock = lock
        self.index_calls = []
        self.lock_taken = False
        self.released = []

    # --- qdrant ---
    def list_collections(self, ttl=30, refresh=False):
        return self.collections

    def collection_info(self, name):
        if name not in self.collections:
            return None
        return {"points_count": self.points.get(name, 0)}

    def point_count(self, name):
        return self.points.get(name, 0)

    def read_lock(self, tenant_id):
        return self.lock

    def take_lock(self, tenant_id, ttl):
        if self.lock and self.lock.get("state") == "running":
            return False
        self.lock_taken = True
        return True

    def release_lock(self, tenant_id, state, detail=None):
        self.released.append((state, detail))

    # --- tenants ---
    def lookup(self, tenant_id):
        return self.tenant

    def collection_on_page(self, page_id):
        return self.page_collection

    # --- indexer ---
    def index_products(self, **kwargs):
        self.index_calls.append(kwargs)
        return {"indexed": 7, "skipped": 0, "processed": 7}


def install(stub, monkey):
    """Point the module's collaborators at the stub."""
    monkey(qdrant, "list_collections", stub.list_collections)
    monkey(qdrant, "collection_info", stub.collection_info)
    monkey(qdrant, "point_count", stub.point_count)
    monkey(qdrant, "read_lock", stub.read_lock)
    monkey(qdrant, "take_lock", stub.take_lock)
    monkey(qdrant, "release_lock", stub.release_lock)
    monkey(tenants, "lookup", stub.lookup)
    monkey(tenants, "collection_on_page", stub.collection_on_page)


class Patcher:
    """Minimal monkeypatch that puts everything back."""

    def __init__(self):
        self.saved = []

    def __call__(self, obj, name, value):
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)
        self.saved = []


def run(stub, **kwargs):
    """bootstrap_if_missing against a stub, with the indexer intercepted."""
    import types

    patch = Patcher()
    install(stub, patch)

    # bootstrap imports the indexer inside the function, so a fake module in
    # sys.modules is what intercepts it.
    fake_indexer = types.ModuleType("imagesearch.indexer")
    fake_indexer.index_products = stub.index_products
    fake_catalog = types.ModuleType("imagesearch.catalog")

    class FakeClient:
        base_url = "https://example.test/api"

        def __init__(self, *a, **k):
            pass

        def reachable(self):
            return True

    fake_catalog.CatalogClient = FakeClient
    fake_catalog.pull_products = lambda client: iter([{"sku": "A", "id": 1}])

    real = {k: sys.modules.get(k) for k in ("imagesearch.indexer", "imagesearch.catalog")}
    sys.modules["imagesearch.indexer"] = fake_indexer
    sys.modules["imagesearch.catalog"] = fake_catalog
    try:
        return bootstrap.bootstrap_if_missing("tn_1_2", **kwargs)
    finally:
        for k, v in real.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        patch.undo()


TENANT = tenants.Tenant(
    ai_tenant_id="tn_1_2",
    laravel_base_url="https://example.test/api",
    laravel_token="tok",
    page_id="900",
)


def main():
    print("An indexed tenant is never re-embedded")
    stub = Stub(collections=["tenant_products_tn_1_2"],
                points={"tenant_products_tn_1_2": 2208},
                tenant=TENANT)
    out = run(stub)
    check("status is skipped", out["status"], "skipped")
    check("nothing was indexed", stub.index_calls, [])
    check("the existing collection is reported",
          out["collection"], "tenant_products_tn_1_2")

    print("\nA rebuilt tenant is recognised through its stamped collection")
    stub = Stub(collections=["tenant_products_tn_1_2",
                             "tenant_products_tn_1_2_1787171658"],
                points={"tenant_products_tn_1_2": 4,
                        "tenant_products_tn_1_2_1787171658": 2208},
                tenant=tenants.Tenant("tn_1_2", "u", "t", page_id=None))
    out = run(stub)
    check("status is skipped", out["status"], "skipped")
    check("the newest rebuild is chosen",
          out["collection"], "tenant_products_tn_1_2_1787171658")
    check("nothing was indexed", stub.index_calls, [])

    print("\nThe page row wins over the derived name")
    stub = Stub(collections=["tenant_products_tn_1_2",
                             "tenant_products_tn_1_2_1787171658"],
                points={"tenant_products_tn_1_2": 1,
                        "tenant_products_tn_1_2_1787171658": 9},
                page_collection="tenant_products_tn_1_2",
                tenant=TENANT)
    patch = Patcher()
    install(stub, patch)
    try:
        r = bootstrap.resolve("tn_1_2")
    finally:
        patch.undo()
    check("resolves to what the page names", r.collection, "tenant_products_tn_1_2")
    check("source is the page row", r.source, "page_row")

    print("\nAn unindexed tenant IS embedded")
    stub = Stub(collections=[], tenant=TENANT)
    out = run(stub)
    check("status is completed", out["status"], "completed")
    check("the indexer ran once", len(stub.index_calls), 1)
    check("into the tenant's own collection",
          stub.index_calls[0]["collection_name"], "tenant_products_tn_1_2")
    check("carrying the tenant id", stub.index_calls[0]["ai_tenant_id"], "tn_1_2")
    check("and the page id", stub.index_calls[0]["page_id"], "900")
    check("the lock was released as completed",
          stub.released, [("completed", None)])

    print("\nAn empty collection counts as unindexed, and is reused not renamed")
    stub = Stub(collections=["tenant_products_tn_1_2"],
                points={"tenant_products_tn_1_2": 0},
                tenant=TENANT)
    out = run(stub)
    check("status is completed", out["status"], "completed")
    check("indexed into the collection that was already there",
          stub.index_calls[0]["collection_name"], "tenant_products_tn_1_2")

    print("\nA bootstrap already running is not started twice")
    stub = Stub(collections=[], tenant=TENANT,
                lock={"state": "running", "started_at": 9e18})
    out = run(stub)
    check("status is locked", out["status"], "locked")
    check("nothing was indexed", stub.index_calls, [])

    print("\nforce is refused unless the service allows it")
    stub = Stub(collections=["tenant_products_tn_1_2"],
                points={"tenant_products_tn_1_2": 2208},
                tenant=TENANT)
    out = run(stub, force=True)
    check("status is failed", out["status"], "failed")
    check("nothing was indexed", stub.index_calls, [])
    check("and it says why", "ALLOW_FORCE_REINDEX" in out["detail"], True)

    print("\nforce re-embeds once the service allows it")
    stub = Stub(collections=["tenant_products_tn_1_2"],
                points={"tenant_products_tn_1_2": 2208},
                tenant=TENANT)
    patch = Patcher()
    patch(bootstrap, "ALLOW_FORCE_REINDEX", True)
    try:
        out = run(stub, force=True)
    finally:
        patch.undo()
    check("status is completed", out["status"], "completed")
    check("the indexer ran", len(stub.index_calls), 1)
    check("with force_reindex set", stub.index_calls[0]["force_reindex"], True)

    print("\nUnreachable credentials fail before anything is created")
    stub = Stub(collections=[], tenant=None)
    out = run(stub)
    check("status is failed", out["status"], "failed")
    check("nothing was indexed", stub.index_calls, [])
    check("and it names the tenant", "tn_1_2" in out["detail"], True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
