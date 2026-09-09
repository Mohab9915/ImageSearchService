"""Turning a catalog into vectors.

MIRROR. A verbatim copy of the chatbot's `helpers/product_image_indexer.py`,
with its two imports repointed at this package. Copied rather than rewritten
on purpose: both services write into the *same* collections, so the payload
keys, the point ids and the choice of which pictures to index have to be
identical. A cleaner reimplementation here would produce points the chatbot's
search reads as missing their colour, sizes and price.

Keep in step with the original. The parts that must not drift are
`build_product_payload`, `_point_id`, `product_images` and `colour_images`.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

import httpx

from imagesearch.config import QDRANT_API_KEY, QDRANT_URL
from imagesearch.embedding import get_embedding_model

logger = logging.getLogger(__name__)

VECTOR_SIZE = 768
BATCH_SIZE = 50

# Tuning knobs, all overridable on the service without a code change.
#
# The deadline is the important one. Each product carries up to three
# candidate image URLs, and a product whose images are all dead used to cost
# three full read timeouts -- 30 seconds of a worker doing nothing. On a
# catalog with many broken images that, not the model, is what sets the pace.
INDEX_WORKERS = int(os.environ.get("INDEX_WORKERS", "64"))
INDEX_EMBED_BATCH = int(os.environ.get("INDEX_EMBED_BATCH", "64"))

# How many batches to embed at once. One forward pass given every core does
# not use them well -- a ViT at batch 64 stops scaling somewhere around 8-16
# threads -- so several workers on a slice of the cores each beat one worker
# on all of them. torch releases the GIL for the forward pass, so these
# genuinely overlap rather than taking turns.
EMBED_WORKERS = int(os.environ.get("INDEX_EMBED_WORKERS", "4"))
IMAGE_READ_TIMEOUT = float(os.environ.get("INDEX_IMAGE_TIMEOUT", "6"))
IMAGE_CONNECT_TIMEOUT = float(os.environ.get("INDEX_IMAGE_CONNECT_TIMEOUT", "3"))
PRODUCT_DEADLINE = float(os.environ.get("INDEX_PRODUCT_DEADLINE", "12"))

# How many of a product's pictures to index. Customers photograph the item
# rather than the listing's first shot, so indexing only that one means the
# photo they send matches nothing. Each extra picture is another embedding,
# so this is what indexing time scales with.
INDEX_MAX_IMAGES = int(os.environ.get("INDEX_MAX_IMAGES", "4"))

# How often to report progress. A full catalog sync is the longest-running
# thing this service does, and it used to log nothing at all between "queued"
# and "finished" -- so a job that was working normally was indistinguishable
# from one that had died.
PROGRESS_EVERY = int(os.environ.get("INDEX_PROGRESS_EVERY", "200"))

def tenant_product_collection_name(ai_tenant_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", ai_tenant_id or "").strip("_").lower()
    return f"tenant_products_{safe or 'unknown'}"

def qdrant_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    return headers

def ensure_collection(collection_name: str) -> None:
    exists = httpx.get(
        f"{QDRANT_URL}/collections/{collection_name}",
        headers=qdrant_headers(),
        timeout=10,
    )
    if exists.status_code == 200:
        return

    payload = {"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}}
    resp = httpx.put(
        f"{QDRANT_URL}/collections/{collection_name}",
        json=payload,
        headers=qdrant_headers(),
        timeout=20,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create Qdrant collection {collection_name}: {resp.status_code} {resp.text[:300]}")
    logger.info("[QDRANT] Created tenant product collection %s", collection_name)

def delete_product(collection_name: str, identifier: str) -> None:
    """Remove a product by sku or by product id.

    A delete push carries only `{"id": 123}`, so the sku the point was
    stored under is not available to derive its point id from.
    """
    should = [
        {"key": "sku", "match": {"value": str(identifier)}},
        {"key": "product_id", "match": {"value": str(identifier)}},
    ]
    if str(identifier).isdigit():
        should.append({"key": "product_id", "match": {"value": int(identifier)}})

    resp = httpx.post(
        f"{QDRANT_URL}/collections/{collection_name}/points/delete",
        json={"filter": {"should": should}},
        headers=qdrant_headers(),
        timeout=15,
    )
    if resp.status_code not in (200, 202):
        logger.warning("[QDRANT] Product delete failed for %s/%s: %s",
                       collection_name, identifier, resp.text[:200])

def delete_collection(collection_name: str) -> bool:
    """Drop a tenant's collection entirely.

    Called when a tenant disconnects. Collection names are derived from
    ai_tenant_id, so leaving the vectors behind means a later reconnect
    reuses the same collection and compares against fingerprints belonging
    to the previous tenancy -- which reads as "already indexed" and skips
    the whole catalog. Dropping it here is what makes a reconnect start
    from a genuinely clean slate.
    """
    try:
        resp = httpx.delete(
            f"{QDRANT_URL}/collections/{collection_name}",
            headers=qdrant_headers(),
            timeout=30,
        )
        # 404 is success for our purposes: the collection is gone either way.
        if resp.status_code in (200, 202, 404):
            logger.info("[QDRANT] Dropped collection %s", collection_name)
            return True
        logger.warning("[QDRANT] Could not drop %s: %s %s",
                       collection_name, resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("[QDRANT] Could not drop %s: %s", collection_name, e)
    return False


def list_tenant_collections(base_name: str) -> List[str]:
    """Every collection belonging to one tenant: `base` and `base_<stamp>`.

    Matched exactly rather than by prefix, because tenant ids are not
    prefix-free -- `tenant_products_tn_2_3` is a prefix of
    `tenant_products_tn_2_30`, a different tenant entirely.
    """
    try:
        resp = httpx.get(f"{QDRANT_URL}/collections", headers=qdrant_headers(), timeout=20)
        if resp.status_code != 200:
            return []
        names = [c.get("name", "") for c in resp.json()["result"]["collections"]]
    except Exception as e:
        logger.warning("[QDRANT] Could not list collections: %s", e)
        return []

    return [n for n in names if n == base_name or n.startswith(base_name + "_")]


def set_point_payload(collection_name: str, point_id: str, payload: Dict[str, Any]) -> None:
    """Rewrite one point's payload, leaving its vector alone.

    A price or stock change does not move the image, and the vector is a
    function of the image alone -- so re-downloading and re-embedding to
    apply one is pure waste. Measured on a live tenant, a single price edit
    arrived as 8 separate push events and cost 8 image downloads and 8
    forward passes at 250-490ms each; as payload writes it is 8 small
    HTTP calls.
    """
    resp = httpx.post(
        f"{QDRANT_URL}/collections/{collection_name}/points/payload?wait=false",
        json={"payload": payload, "points": [point_id]},
        headers=qdrant_headers(),
        timeout=30,
    )
    if resp.status_code not in (200, 202):
        raise RuntimeError(
            f"Qdrant payload update failed for {collection_name}: "
            f"{resp.status_code} {resp.text[:200]}"
        )


def get_existing_points(collection_name: str) -> Dict[str, Dict[str, Any]]:
    """Map point id -> {updated_at, image_url, price} for everything indexed.

    The first two answer different questions. Together they say whether a
    product changed at all, so a reconciliation can skip it. The image URL
    alone says whether the *vector* is still valid, so a product that
    changed in some other way can have its payload rewritten without going
    near the model. Price is carried only so the logs can show what a sync
    actually changed.
    """
    fingerprints: Dict[str, Dict[str, Any]] = {}
    next_offset = None
    headers = qdrant_headers()
    try:
        while True:
            payload = {
                "limit": 10000,
                # only the two fields the fingerprint needs, not whole payloads
                "with_payload": ["updated_at", "image_url", "price"],
                "with_vector": False,
            }
            if next_offset:
                payload["offset"] = next_offset

            resp = httpx.post(
                f"{QDRANT_URL}/collections/{collection_name}/points/scroll",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            if resp.status_code != 200:
                logger.warning("[QDRANT] Scroll failed for %s: %s %s",
                               collection_name, resp.status_code, resp.text[:100])
                break

            res = resp.json().get("result", {})
            points = res.get("points", [])
            for p in points:
                pid = p.get("id")
                if not pid:
                    continue
                pl = p.get("payload") or {}
                fingerprints[str(pid)] = {
                    "updated_at": str(pl.get("updated_at") or ""),
                    "image_url": str(pl.get("image_url") or ""),
                    "price": pl.get("price"),
                }

            next_offset = res.get("next_page_offset")
            if not next_offset or not points:
                break
        logger.debug("[QDRANT] %s already holds %d points",
                     collection_name, len(fingerprints))
    except Exception as e:
        logger.warning("[QDRANT] Error reading existing points from %s: %s",
                       collection_name, e)
    return fingerprints


def index_products(
    *,
    collection_name: str,
    products: Iterable[Dict[str, Any]],
    ai_tenant_id: str,
    page_id: Optional[str] = None,
    limit: int = 0,
    workers: int = INDEX_WORKERS,
    embed_batch_size: int = INDEX_EMBED_BATCH,
    force_reindex: bool = False,
    allow_skip: bool = True,
    explain: bool = False,
    default_currency: Optional[str] = None,
    locale: Optional[str] = None,
    progress: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait

    ensure_collection(collection_name)

    existing = {} if force_reindex else get_existing_points(collection_name)

    # Split the cores across the embed workers rather than giving every
    # worker the whole machine, which would just oversubscribe.
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // max(1, EMBED_WORKERS)))
    except Exception:
        pass

    model, processor = get_embedding_model()

    # One pooled client for every download. httpx.get() opens a fresh
    # connection per call, so a 30k catalog meant 30k TCP and TLS
    # handshakes against the same few hosts.
    http = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (ImageSearch/1.0)"},
        timeout=httpx.Timeout(IMAGE_READ_TIMEOUT, connect=IMAGE_CONNECT_TIMEOUT),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=workers,
                            max_keepalive_connections=workers),
    )

    def _download(product: Dict[str, Any], wanted: List[tuple]):
        """Fetch each requested image, preprocessing it here.

        Preprocessing runs in this worker rather than in the embed step so
        the resize and normalise cost -- a large share of the CPU work --
        is spread across the pool instead of being serialised in front of
        the model.
        """
        sku = _product_sku(product)
        if not sku:
            return []

        # Stop trying alternates once this product has had its share of the
        # clock, so one unreachable origin cannot hold a worker hostage.
        deadline = time.monotonic() + PRODUCT_DEADLINE
        results = []

        for key, group, extra in wanted:
            if time.monotonic() >= deadline:
                logger.debug("[QDRANT] %s: image deadline hit after %d image(s)", sku, len(results))
                break

            for image_url in group[:3]:
                try:
                    resp = http.get(image_url)
                    if resp.status_code >= 400:
                        continue
                    if not resp.headers.get("content-type", "").startswith("image/"):
                        continue
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                    # Downscale before preprocessing: the model works at
                    # 224px, and holding full-resolution RGB for every
                    # in-flight product is what exhausted memory on big
                    # catalogs.
                    img.thumbnail((512, 512), Image.LANCZOS)
                    if processor is not None:
                        try:
                            img = processor(images=img, return_tensors="pt")["pixel_values"][0]
                        except Exception:
                            pass
                    results.append((product, sku, image_url, img, key, extra))
                    break
                except Exception:
                    continue

        if not results:
            first = wanted[0][1][0] if wanted and wanted[0][1] else ""
            results.append((product, sku, first, None, "", {}))
        return results

    def _embed_batch(batch: List[tuple]) -> List[dict]:

        if not batch:
            return []
        if not model or not processor:
            return []
        items = [item[3] for item in batch]
        try:
            # inference_mode is cheaper than no_grad: it also skips version
            # counter bookkeeping on every tensor.
            with torch.inference_mode():
                if torch.is_tensor(items[0]):
                    inputs = {"pixel_values": torch.stack(items)}
                else:
                    inputs = processor(images=items, return_tensors="pt", padding=True)
                features = model.get_image_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    po = getattr(features, 'pooler_output', None)
                    if po is not None:
                        features = po
                    else:
                        features = features.last_hidden_state[:, 0]
                embeddings = F.normalize(features, p=2, dim=-1).tolist()
        except Exception as e:
            logger.warning("[QDRANT] Batch embedding failed: %s", e)
            return []
        return [
            {
                "id": _point_id(collection_name, sku, key),
                "vector": vector,
                "payload": build_product_payload(product, sku, image_url, ai_tenant_id,
                                                 page_id, default_currency, locale, extra),
            }
            for (product, sku, image_url, _, key, extra), vector in zip(batch, embeddings)
        ]

    # Qdrant writes leave the critical path. The main thread is the only one
    # that consumes finished downloads and feeds the model, so every second
    # it spent blocked on an upsert round trip was a second the download pool
    # and the model both sat idle.
    writers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qdrant-write")
    write_futures: deque = deque()

    def _reap_writes(keep: int = 0) -> None:
        """Surface write failures and bound how far ahead writes may queue."""
        while len(write_futures) > keep:
            write_futures.popleft().result()

    def _upsert_buffer(buf: List[dict]) -> int:

        total = 0
        for i in range(0, len(buf), BATCH_SIZE):
            chunk = buf[i:i + BATCH_SIZE]
            write_futures.append(writers.submit(upsert_points, collection_name, chunk))
            total += len(chunk)
        # Backpressure: if Qdrant falls behind, block here rather than let
        # finished points accumulate in memory without limit.
        _reap_writes(keep=32)
        return total

    payload_writes: List[tuple] = []

    def _flush_payload_writes(*, keep: int = 0) -> None:
        """Send buffered payload-only updates. One point per call: Qdrant
        sets one payload across many points, not many payloads at once."""
        while len(payload_writes) > keep:
            point_id, payload = payload_writes.pop(0)
            write_futures.append(
                writers.submit(set_point_payload, collection_name, point_id, payload)
            )
        _reap_writes(keep=32)

    indexed = 0
    skipped = 0
    repayloaded = 0
    count = 0
    pending: List[tuple] = []
    upsert_buf: List[dict] = []

    # Where the wall clock actually goes. The main thread is the pipeline's
    # only consumer, so attributing its time tells us which stage is the
    # constraint: t_stall means downloads are too slow to keep the model fed,
    # t_embed means the CPU is the wall, t_write means Qdrant is.
    t_start = time.monotonic()
    t_embed = 0.0
    t_write = 0.0
    t_stall = 0.0
    embedded = 0
    _embed_lock = threading.Lock()

    # Embedding runs off the main thread now, so several batches are in the
    # model at once instead of the pipeline stopping for each one.
    embed_pool = ThreadPoolExecutor(max_workers=EMBED_WORKERS, thread_name_prefix="embed")
    embed_futures: deque = deque()

    def _timed_embed(batch: List[tuple]):
        nonlocal t_embed, embedded
        started = time.monotonic()
        points = _embed_batch(batch)
        elapsed = time.monotonic() - started
        with _embed_lock:
            t_embed += elapsed
            embedded += len(batch)
        return points, len(batch)

    def _collect_embeds(*, drain_all: bool) -> None:
        """Fold finished batches into the write buffer.

        Blocks on the oldest batch once the backlog exceeds what the pool can
        be working on, which is the backpressure that stops finished
        downloads piling up faster than the model consumes them.
        """
        nonlocal indexed, skipped
        limit = 0 if drain_all else EMBED_WORKERS * 2
        while len(embed_futures) > limit:
            points, n = embed_futures.popleft().result()
            if points:
                upsert_buf.extend(points)
                indexed += len(points)
            else:
                skipped += n

    def _drain(results) -> None:
        """Fold a product's downloaded images into the embed/upsert pipeline."""
        nonlocal indexed, t_write

        for result in results or []:
            product, sku, image_url, img, key, extra = result
            if img is None:
                # No usable image: still index the product so text search and
                # SKU lookups find it, with a zero vector it can never match on.
                upsert_buf.append({
                    "id": _point_id(collection_name, sku, key),
                    "vector": [0.0] * VECTOR_SIZE,
                    "payload": build_product_payload(product, sku, image_url, ai_tenant_id,
                                                     page_id, default_currency, locale, extra),
                })
                indexed += 1
            else:
                pending.append(result)

        while len(pending) >= embed_batch_size:
            batch = pending[:embed_batch_size]
            del pending[:embed_batch_size]
            embed_futures.append(embed_pool.submit(_timed_embed, batch))

        _collect_embeds(drain_all=False)

        while len(upsert_buf) >= BATCH_SIZE:
            _t = time.monotonic()
            _upsert_buffer(upsert_buf[:BATCH_SIZE])
            t_write += time.monotonic() - _t
            del upsert_buf[:BATCH_SIZE]

    # Bounded window: keep at most `window` downloads in flight and consume
    # each as it lands. The previous version submitted every product to the
    # pool up front, so a 30k-product catalog queued 30k tasks and let
    # decoded RGB images pile up faster than they were embedded -- gigabytes
    # of resident memory. Peak memory is now a function of the window, not
    # of catalog size.
    window = max(workers * 2, embed_batch_size)

    # try/finally, because the product source can raise mid-run -- a catalog
    # page that will not load aborts the pull -- and without it every failed
    # sync would strand the embed and writer pools and their threads.
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            in_flight = set()

            for product in products:
                if limit and count >= limit:
                    break

                sku = _product_sku(product)
                incoming_updated = str(product.get("updated_at") or "")

                # The product's own pictures, then one per colour so a photo
                # tells the bot which colour it is looking at.
                shots = [("" if n == 0 else group[0], group, {})
                         for n, group in enumerate(product_images(product)[:INDEX_MAX_IMAGES])]
                shots += [(group[0], group, extra)
                          for group, extra in colour_images(product)]
                if not shots:
                    shots = [("", [""], {})]

                # Decided per picture: a product whose first photo is
                # untouched can still have gained a colour, and that one has
                # to be fetched.
                wanted = []
                for key, group, extra in shots:
                    image_url = group[0]
                    prior = existing.get(
                        _point_id(collection_name, sku, key)
                    ) if (existing and sku) else None

                    if force_reindex or not prior:
                        wanted.append((key, group, extra))
                    elif (allow_skip
                          and prior["updated_at"] == incoming_updated
                          and prior["image_url"] == image_url):
                        skipped += 1
                    elif image_url and prior["image_url"] == image_url:
                        payload = build_product_payload(
                            product, sku, image_url, ai_tenant_id, page_id,
                            default_currency, locale, extra)
                        if explain:
                            logger.info(
                                "[INDEX] %s PAYLOAD-ONLY %s - price %s -> %s, stock %s",
                                sku, extra.get("colour") or "main",
                                prior["price"], payload.get("price"),
                                payload.get("colour_in_stock", payload.get("in_stock")),
                            )
                        payload_writes.append((_point_id(collection_name, sku, key), payload))
                        repayloaded += 1
                        _flush_payload_writes(keep=64)
                    else:
                        wanted.append((key, group, extra))

                if not wanted:
                    if explain:
                        logger.info("[INDEX] %s nothing to fetch", sku)
                    continue

                if explain:
                    colours = [e.get("colour") for _, _, e in wanted if e.get("colour")]
                    logger.info("[INDEX] %s fetching %d of %d picture(s)%s",
                                sku, len(wanted), len(shots),
                                f" colours={colours}" if colours else "")

                count += 1
                if progress is not None:
                    progress["processed"] = count
                in_flight.add(pool.submit(_download, product, wanted))

                if PROGRESS_EVERY and count % PROGRESS_EVERY == 0:
                    so_far = max(time.monotonic() - t_start, 1e-6)
                    logger.info(
                        "[INDEX] %s: %d products, %.1f/s (embed %.0fms each)",
                        collection_name, count, count / so_far,
                        (t_embed / embedded * 1000) if embedded else 0.0,
                    )

                if len(in_flight) >= window:
                    _t = time.monotonic()
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    t_stall += time.monotonic() - _t
                    for future in done:
                        try:
                            _drain(future.result())
                        except Exception as e:
                            logger.warning("[QDRANT] Product download failed: %s", e)
                            skipped += 1

            for future in as_completed(in_flight):
                try:
                    _drain(future.result())
                except Exception as e:
                    logger.warning("[QDRANT] Product download failed: %s", e)
                    skipped += 1

        # Whatever did not fill a full batch, then everything still in the
        # model, then everything still queued for Qdrant.
        if pending:
            embed_futures.append(embed_pool.submit(_timed_embed, list(pending)))
            pending.clear()
        _collect_embeds(drain_all=True)

        if upsert_buf:
            _upsert_buffer(upsert_buf)
        _flush_payload_writes()
        _reap_writes()
    finally:
        embed_pool.shutdown(wait=True)
        writers.shutdown(wait=True)
        http.close()

    elapsed = max(time.monotonic() - t_start, 1e-6)
    logger.info(
        "[INDEX] %s: %d products in %.0fs (%.1f/s) on %d cpu | download-stall %.0fs  "
        "embed %.0fs (%d imgs, %.0fms each)  qdrant %.0fs  payload-only %d",
        collection_name, count, elapsed, count / elapsed, os.cpu_count() or 0,
        t_stall, t_embed, embedded,
        (t_embed / embedded * 1000) if embedded else 0.0,
        t_write, repayloaded,
    )
    return {
        "indexed": indexed + repayloaded,
        "skipped": skipped,
        "processed": count,
        "repayloaded": repayloaded,
    }


def upsert_points(collection_name: str, points: List[dict]) -> None:
    resp = httpx.put(
        # wait=false: return once the write is accepted rather than once it
        # is searchable. The sync does not need per-batch read-after-write,
        # and waiting serialises the pipeline behind Qdrant's indexer.
        f"{QDRANT_URL}/collections/{collection_name}/points?wait=false",
        json={"points": points},
        headers=qdrant_headers(),
        timeout=45,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Qdrant upsert failed for {collection_name}: {resp.status_code} {resp.text[:300]}")

_SIZE_KEYS = ("original_image_url", "url", "large_image_url",
              "medium_image_url", "small_image_url", "path")


def product_images(product: Dict[str, Any]) -> List[List[str]]:
    """One entry per distinct picture, each listing that picture's sizes.

    A customer photographs the item, not the listing's first shot, so every
    picture is indexed. Sizes of the same picture are grouped rather than
    listed side by side: they are one image to search for and several URLs
    to try if a download fails.
    """
    groups: List[List[str]] = []

    def sizes(source: Dict[str, Any]) -> List[str]:
        out = []
        for key in _SIZE_KEYS:
            value = source.get(key)
            if value and isinstance(value, str) and value not in out:
                out.append(value)
        return out

    base = product.get("base_image")
    if isinstance(base, dict):
        groups.append(sizes(base))

    for image in (product.get("images") or []):
        if isinstance(image, str):
            groups.append([image])
        elif isinstance(image, dict):
            groups.append(sizes(image))

    for key in ("thumbnail", "image_url", "image"):
        value = product.get(key)
        if value and isinstance(value, str):
            groups.append([value])

    # Keyed on the filename, not the URL. The same picture is served from
    # both the storage host and the cache host, so comparing URLs counts it
    # twice and spends one of the slots on a duplicate.
    merged: Dict[str, List[str]] = {}
    for group in groups:
        if not group:
            continue
        key = group[0].rsplit("/", 1)[-1] or group[0]
        for url in group:
            if url not in merged.setdefault(key, []):
                merged[key].append(url)
    return list(merged.values())


def image_url_candidates(product: Dict[str, Any]) -> List[str]:
    """Every plausible image URL for a product, best first."""
    return [url for group in product_images(product) for url in group]


def _variant_in_stock(variant: Dict[str, Any]) -> bool:
    for source in (variant, variant.get("stock") or {}):
        if isinstance(source, dict) and "in_stock" in source:
            return bool(source["in_stock"])
    return True


def colour_images(product: Dict[str, Any]) -> List[tuple]:
    """One picture per colour, with what the colour is and what is in stock.

    Bagisto marks the visual attribute by giving its options a swatch
    colour, and sizes have none -- so colours are identified without
    guessing at attribute names, which are whatever the merchant typed.
    """
    data = product.get("configurable_data") or {}
    attributes = data.get("attributes") or {}
    matrix = data.get("variant_matrix") or []
    if not matrix:
        return []

    option_label = {
        (code, o.get("id")): str(o.get("label") or "")
        for code, a in attributes.items()
        for o in (a.get("options") or [])
    }

    out = []
    for code, attribute in attributes.items():
        options = attribute.get("options") or []
        if not any(o.get("swatch_value") for o in options):
            continue

        for option in options:
            option_id = option.get("id")
            variants = [v for v in matrix
                        if (v.get("attributes") or {}).get(code) == option_id]
            if not variants:
                continue

            stocked = [v for v in variants if _variant_in_stock(v)]
            representative = (stocked or variants)[0]
            groups = product_images(representative)
            if not groups:
                continue

            price, currency = _extract_price(representative)

            sizes = sorted({
                option_label.get((c, o), "")
                for v in stocked
                for c, o in (v.get("attributes") or {}).items()
                if c != code
            } - {""})

            extra = {
                "colour": str(option.get("label") or ""),
                "colour_swatch": option.get("swatch_value"),
                "colour_attribute": attribute.get("label") or code,
                "colour_in_stock": bool(stocked),
                "colour_sizes": sizes,
                "variant_id": representative.get("id"),
            }
            # build_product_payload applies extra last, so this wins over the
            # parent's. Left out entirely when the variant has no price, so
            # the parent's still shows rather than nothing.
            if price is not None:
                extra["price"] = price
                if currency:
                    extra["currency"] = currency
            out.append((groups[0], extra))
    return out



def _extract_price(product: Dict[str, Any]):
    """Price as a number, plus its currency.

    The documented push shape nests it -- {"regular": 49.99, "special":
    39.99, "currency": "JOD"} -- while a product pulled from the API may
    carry a bare number or a preformatted string. Reading only the flat key
    stored the whole dict as the price, so the bot quoted a dict at
    customers. Special beats regular, because that is what they pay.
    """
    import re as _re

    raw = product.get("price")
    currency = None

    if isinstance(raw, dict):
        currency = raw.get("currency")
        for key in ("special", "final", "regular", "amount", "value"):
            val = raw.get(key)
            if val not in (None, "", 0):
                raw = val
                break
        else:
            raw = None

    if isinstance(raw, (int, float)):
        return float(raw), currency

    for candidate in (raw,
                      product.get("formatted_special_price"),
                      product.get("formatted_price"),
                      product.get("special_price")):
        if candidate in (None, ""):
            continue
        if isinstance(candidate, (int, float)):
            return float(candidate), currency
        text = str(candidate)
        m = _re.search(r"[\d]+(?:[.,]\d+)?", text.replace(",", ""))
        if m:
            try:
                # The only place the currency appears on this backend is the
                # symbol in front of the number -- there is no price object
                # and no currency field -- so read it rather than discard it
                # and let the bot fall back to a hardcoded one.
                return float(m.group().replace(",", "")), currency or _symbol_currency(text)
            except ValueError:
                continue

    return None, currency


def _symbol_currency(text: str) -> Optional[str]:
    """Currency code from a preformatted price like "₪200.00"."""
    from imagesearch.money import currency_from_text

    return currency_from_text(text)


def _extract_stock(product: Dict[str, Any]):
    """(in_stock, qty), accepting the nested {"stock": {...}} shape."""
    stock = product.get("stock")
    if isinstance(stock, dict):
        return stock.get("in_stock"), stock.get("qty")
    return product.get("in_stock"), product.get("qty")


def translated(product: Dict[str, Any], locale: Optional[str] = None) -> Dict[str, Any]:
    """The locale's strings. Product pushes nest them; pulled products do not."""
    translations = product.get("translations")
    if not isinstance(translations, dict) or not translations:
        return product
    return (translations.get(locale)
            or translations.get("en")
            or next(iter(translations.values()), {}))


def build_product_payload(
    product: Dict[str, Any],
    sku: str,
    image_url: str,
    ai_tenant_id: str,
    page_id: Optional[str],
    default_currency: Optional[str] = None,
    locale: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = translated(product, locale)
    name = text.get("name") or text.get("title") or ""
    price, currency = _extract_price(product)
    in_stock, qty = _extract_stock(product)

    payload = {
        "source": "laravel_catalog",
        "ai_tenant_id": ai_tenant_id,
        "page_id": page_id,
        "sku": sku,
        "product_sku": sku,
        "product_id": product.get("id"),
        "name": name,
        "title": name,
        "description": text.get("description") or text.get("short_description") or "",
        "image_url": image_url,
        "price": price,
        "currency": currency or default_currency,
        "type": product.get("type", "simple"),
        "in_stock": in_stock,
        "qty": qty,
        # Read back by get_existing_points to decide what needs re-embedding.
        "updated_at": product.get("updated_at"),
        "url": text.get("url") or product.get("product_url"),
        "raw_product": product,
    }
    payload.update(extra or {})
    return payload

def _product_sku(product: Dict[str, Any]) -> str:

    return str(
        product.get("sku") or
        product.get("url_key") or
        product.get("id") or
        ""
    ).strip()

def _point_id(collection_name: str, sku: str, image_url: str = "") -> str:
    """The product's own point, or one of its extra images.

    The first image keeps the bare `collection:sku` id so a lookup by SKU
    still finds the product without knowing which images it has.
    """
    key = f"{collection_name}:{sku}"
    if image_url:
        key = f"{key}:{image_url}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
