"""Search a tenant's catalog by picture.

A separate service from the chatbot, sharing its Qdrant. It searches the
collections the chatbot has already built, and indexes a tenant only when
that tenant has no collection at all.

    GET  /health
    GET  /tenants/{ai_tenant_id}              what collection, how many points
    POST /tenants/{ai_tenant_id}/bootstrap    index, if and only if unindexed
    POST /tenants/{ai_tenant_id}/events/product   one product created/updated/deleted
    POST /search                              by image url, or an upload
    POST /search/text                         by phrase, same vectors
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException,
    Query, Request, UploadFile,
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from imagesearch import bootstrap, embedding, events, qdrant, results
from imagesearch.config import (
    API_TOKEN, DEFAULT_LIMIT, DEFAULT_SCORE_THRESHOLD, MAX_UPLOAD_BYTES,
    QDRANT_URL, RATE_LIMIT_PER_MINUTE, SIGLIP_MODEL, SIMILAR_SCORE_THRESHOLD,
)
from imagesearch.ratelimit import RateLimiter

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("imagesearch")

# httpx logs a line per request at INFO. A catalog bootstrap makes tens of
# thousands of them, which buries everything worth reading.
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Image Search Service", version="1.0.0")


# ------------------------------------------------------------------- auth
bearer_scheme = HTTPBearer(auto_error=False)


def require_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> None:
    """Bearer token, when one is configured.

    Unset means open. That is fine on a private network and wrong on a public
    URL, which is why /health reports whether it is on.
    """
    if not API_TOKEN:
        return
    supplied = credentials.credentials if credentials else ""
    if supplied != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


# ------------------------------------------------------------ rate limit
_rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    key = request.headers.get("authorization") or (
        request.client.host if request.client else "unknown")
    if not _rate_limiter.allow(key):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded, try again shortly"},
            headers={"Retry-After": "60"},
        )
    return await call_next(request)


# ------------------------------------------------------------------ model
@app.on_event("startup")
def _warmup() -> None:
    """Load the model before the first request.

    Off by default: on Railway the health check has to pass before traffic is
    routed, and loading SigLIP takes longer than the check allows. Set
    WARMUP_ON_START=true where the platform is patient.
    """
    if os.environ.get("WARMUP_ON_START", "false").strip().lower() in ("1", "true", "yes"):
        import threading
        threading.Thread(target=embedding.warmup_model, daemon=True).start()


# ------------------------------------------------------------------ types
class SearchRequest(BaseModel):
    ai_tenant_id: str = Field(..., description="Tenant whose catalog to search")
    image_url: Optional[str] = Field(None, description="Picture to search for")
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=50)
    score_threshold: float = Field(DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0)
    collapse_by_product: bool = Field(
        True,
        description="One row per product rather than one per indexed picture",
    )
    include_similar: bool = True
    similar_threshold: float = Field(SIMILAR_SCORE_THRESHOLD, ge=0.0, le=1.0)


class TextSearchRequest(BaseModel):
    ai_tenant_id: str
    query: str
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=50)
    # Text and image vectors sit in the same space but text scores run lower,
    # so the image threshold would return nothing.
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    collapse_by_product: bool = True


class ProductEvent(BaseModel):
    action: str
    channel_code: Optional[str] = None
    locales: List[str] = Field(default_factory=list)
    product: Dict[str, Any] = Field(default_factory=dict)


class BootstrapRequest(BaseModel):
    laravel_base_url: Optional[str] = Field(
        None, description="Overrides the tenants table; required when it is unavailable")
    laravel_token: Optional[str] = None
    limit: int = Field(0, ge=0, description="Index at most this many products; 0 = all")
    wait: bool = Field(
        False,
        description="Hold the request open until indexing finishes. A full "
                    "catalog takes many minutes, so the default returns "
                    "immediately and you poll GET /tenants/{id}.",
    )
    force: bool = Field(
        False,
        description="Re-embed a tenant that already has a collection. "
                    "Refused unless ALLOW_FORCE_REINDEX is set.",
    )


# ----------------------------------------------------------------- routes
@app.get("/health")
def health() -> Dict[str, Any]:
    collections = qdrant.list_collections()
    return {
        "status": "ok",
        "qdrant_url": QDRANT_URL,
        "qdrant_reachable": bool(collections),
        "collections": len(collections),
        "model": SIGLIP_MODEL,
        "model_loaded": embedding._model is not None,
        "auth_required": bool(API_TOKEN),
    }


@app.get("/tenants/{ai_tenant_id}", dependencies=[Depends(require_token)])
def tenant_status(ai_tenant_id: str) -> Dict[str, Any]:
    """Which collection this tenant resolves to, and whether it is usable."""
    return bootstrap.status(ai_tenant_id)


@app.post("/tenants/{ai_tenant_id}/bootstrap", dependencies=[Depends(require_token)])
def tenant_bootstrap(
    ai_tenant_id: str,
    body: BootstrapRequest,
    background: BackgroundTasks,
) -> Dict[str, Any]:
    """Index this tenant's catalog if -- and only if -- it has none.

    A tenant the chatbot already indexed comes back `skipped` without a
    single embedding being computed.
    """
    kwargs = dict(
        laravel_base_url=body.laravel_base_url,
        laravel_token=body.laravel_token,
        limit=body.limit,
        force=body.force,
    )

    if body.wait:
        return bootstrap.bootstrap_if_missing(ai_tenant_id, **kwargs)

    # Report the decision now even when the work runs in the background --
    # otherwise "accepted" is indistinguishable from "already indexed".
    res = bootstrap.resolve(ai_tenant_id, refresh=True)
    if res.indexed and not body.force:
        return {
            "status": "skipped",
            "ai_tenant_id": ai_tenant_id,
            "collection": res.collection,
            "points": res.points,
            "detail": "tenant already has a collection; nothing was embedded",
        }

    background.add_task(bootstrap.bootstrap_if_missing, ai_tenant_id, **kwargs)
    return {
        "status": "accepted",
        "ai_tenant_id": ai_tenant_id,
        "detail": "indexing started; poll GET /tenants/{id} for progress",
    }


@app.post("/tenants/{ai_tenant_id}/events/product", dependencies=[Depends(require_token)])
def tenant_product_event(
    ai_tenant_id: str,
    body: ProductEvent,
    x_imagesearch_idempotency_key: Optional[str] = Header(
        None, alias="X-ImageSearch-Idempotency-Key"),
) -> Dict[str, Any]:
    return events.handle(ai_tenant_id, x_imagesearch_idempotency_key, body.action, body.product)


def _search(
    ai_tenant_id: str,
    vector: Optional[List[float]],
    limit: int,
    score_threshold: float,
    collapse: bool,
    what: str,
    similar_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    if vector is None:
        raise HTTPException(status_code=422, detail=f"Could not read {what}")

    res = bootstrap.resolve(ai_tenant_id)
    if not res.indexed:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "tenant is not indexed",
                "ai_tenant_id": ai_tenant_id,
                "collection": res.collection,
                "hint": f"POST /tenants/{ai_tenant_id}/bootstrap to index it",
            },
        )

    if similar_threshold is not None:
        raw = qdrant.search(res.collection, vector, limit * 8, similar_threshold)
        rows = results.best_per_product(raw) if collapse else [results.to_row(m) for m in raw]
        exact, similar = results.split_exact_similar(rows, score_threshold, limit)
        return {
            "ai_tenant_id": ai_tenant_id,
            "collection": res.collection,
            "count": len(exact),
            "results": exact,
            "similar_count": len(similar),
            "similar": similar,
        }

    # Ask for more than requested when collapsing: several of the hits will
    # be different pictures of one product and merge into a single row.
    raw = qdrant.search(res.collection, vector,
                        limit * 4 if collapse else limit,
                        score_threshold)

    rows = (results.best_per_product(raw, limit) if collapse
            else [results.to_row(m) for m in raw][:limit])

    return {
        "ai_tenant_id": ai_tenant_id,
        "collection": res.collection,
        "count": len(rows),
        "results": rows,
    }


@app.post("/search", dependencies=[Depends(require_token)])
def search_by_image(body: SearchRequest) -> Dict[str, Any]:
    """Find the products that look like a picture."""
    if not body.image_url:
        raise HTTPException(status_code=422,
                            detail="image_url is required; use /search/upload for a file")
    return _search(
        body.ai_tenant_id,
        embedding.embed_image_url(body.image_url),
        body.limit, body.score_threshold, body.collapse_by_product,
        "the image at that url",
        body.similar_threshold if body.include_similar else None,
    )


@app.post("/search/upload", dependencies=[Depends(require_token)])
async def search_by_upload(
    ai_tenant_id: str = Form(...),
    file: UploadFile = File(...),
    limit: int = Form(DEFAULT_LIMIT),
    score_threshold: float = Form(DEFAULT_SCORE_THRESHOLD),
    collapse_by_product: bool = Form(True),
    include_similar: bool = Form(True),
    similar_threshold: float = Form(SIMILAR_SCORE_THRESHOLD),
) -> Dict[str, Any]:
    """Same search, for a photograph sent straight to us."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"Image larger than {MAX_UPLOAD_BYTES} bytes")
    if not data:
        raise HTTPException(status_code=422, detail="Empty upload")

    return _search(
        ai_tenant_id, embedding.embed_image_bytes(data),
        max(1, min(limit, 50)), score_threshold, collapse_by_product,
        "the uploaded image",
        similar_threshold if include_similar else None,
    )


@app.post("/search/text", dependencies=[Depends(require_token)])
def search_by_text(body: TextSearchRequest) -> Dict[str, Any]:
    """Find products from a phrase, against the same picture vectors."""
    return _search(
        body.ai_tenant_id, embedding.embed_text(body.query),
        body.limit, body.score_threshold, body.collapse_by_product,
        "that query",
    )


@app.get("/collections", dependencies=[Depends(require_token)])
def collections(refresh: bool = Query(False)) -> Dict[str, Any]:
    """Every collection in the shared Qdrant. Diagnostic."""
    return {"collections": sorted(qdrant.list_collections(refresh=refresh))}
