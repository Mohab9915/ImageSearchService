"""A tenant id that prefixes another must not resolve to that tenant's collection."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagesearch import qdrant  # noqa: E402

names = ["tenant_products_tn_2_3", "tenant_products_tn_8_11",
         "tenant_products_tn_24_30_1787756447", "tenant_products_tn_24_30"]
qdrant.list_collections = lambda ttl=30, refresh=False: names

assert qdrant.collection_for_tenant("tn_2") is None
assert qdrant.collection_for_tenant("tn_8") is None
assert qdrant.collection_for_tenant("tn_2_3") == "tenant_products_tn_2_3"
assert qdrant.collection_for_tenant("tn_24_30") == "tenant_products_tn_24_30_1787756447"
print("PASS collection prefix isolation")
