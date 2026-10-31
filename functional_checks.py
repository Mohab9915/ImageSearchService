import asyncio
import math
from io import BytesIO

from health_monitor import install


def install_checks(app):
    async def search():
        def probe():
            import httpx
            from PIL import Image
            from imagesearch import embedding, qdrant
            from imagesearch.config import VECTOR_SIZE, QDRANT_URL
            data = BytesIO()
            Image.new("RGB", (224, 224), "white").save(data, format="PNG")
            vectors = [embedding.embed_image_bytes(data.getvalue()), embedding.embed_text("white cup")]
            for vector in vectors:
                if not vector or len(vector) != VECTOR_SIZE or not all(math.isfinite(v) for v in vector):
                    raise RuntimeError("Embedding failed")
            with httpx.Client(timeout=15, headers=qdrant.headers()) as client:
                response = client.get(f"{QDRANT_URL}/collections")
                response.raise_for_status()
                collections = response.json()["result"]["collections"]
                candidates = [c["name"] for c in collections if c["name"].startswith("tenant_products_")]
                if not candidates:
                    raise RuntimeError("No searchable catalog available")
                for vector in vectors:
                    response = client.post(f"{QDRANT_URL}/collections/{candidates[0]}/points/search",
                                           json={"vector": vector, "limit": 1, "with_payload": False})
                    response.raise_for_status()
                    if not isinstance(response.json()["result"], list):
                        raise RuntimeError("Search failed")
        await asyncio.to_thread(probe)

    async def database():
        def query():
            from imagesearch.tenants import _get_engine
            from sqlalchemy import text
            engine = _get_engine()
            if engine is None:
                raise RuntimeError("Tenant database not configured")
            with engine.connect() as connection:
                connection.execute(text("SELECT ai_tenant_id FROM tenants LIMIT 0"))
        await asyncio.to_thread(query)

    install(app, {"image_and_text_search": search, "tenant_database": database})
