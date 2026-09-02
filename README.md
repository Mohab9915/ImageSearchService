# Image Search Service

Search a merchant's catalog by photograph. A customer sends a picture of a
dress; this answers with the products in that merchant's shop that look like
it, with the colour whose picture actually matched.

It is a **separate service** from the chatbot, and it **shares the chatbot's
Qdrant**. It does not stand up its own vector store, and it does not build
its own copy of a catalog that is already indexed.

## The one rule

> Embed a tenant's products **if and only if** that tenant has no collection
> in Qdrant.

A tenant the chatbot has already indexed is left completely alone. Its
vectors are already there; re-embedding them would spend hours of CPU
producing the same numbers, and doing it in place while the chatbot reads
would serve a half-built catalog to live customers.

`GET /tenants/{id}` tells you which way a tenant will go before you ask for
anything. `tools/check_live.py` prints the decision for every tenant at once,
without writing.

## How it shares the chatbot's index

Three things have to agree, or the two services silently stop understanding
each other's data:

| | Value | Where |
|---|---|---|
| Vector space | SigLIP `google/siglip-base-patch16-224`, 768-dim, L2-normalised, cosine | `imagesearch/embedding.py` |
| Collection naming | `tenant_products_{ai_tenant_id}`, rebuilds as `..._{unix_ts}`, newest wins | `imagesearch/qdrant.py` |
| Point payload and ids | the chatbot's exact shape | `imagesearch/indexer.py` |

`imagesearch/indexer.py` is a **verbatim copy** of the chatbot's
`helpers/product_image_indexer.py` with two imports repointed. It is copied
rather than rewritten because both services write into the same collections:
a cleaner reimplementation here would produce points the chatbot reads as
missing their colour, sizes and price.

`tools/test_mirror_in_step.py` diffs the pinned functions against the
chatbot's copy. Run it after touching either side.

## Resolving a tenant's collection

Two sources, and they can disagree:

1. **The page row** in the chatbot's Postgres (`PagesTable.qdrant_collection_name`).
   This is what the chatbot itself reads through, so it wins.
2. **The collection names in Qdrant** — the newest `_{timestamp}` suffix,
   else the bare name. The fallback for a tenant whose page row was never
   written, and the only source when the service runs without a database.

A page row naming a collection Qdrant no longer holds falls back to (2)
rather than concluding the tenant is unindexed.

An **empty** collection counts as unindexed. It is reused rather than renamed
— the chatbot's page row may already point at it.

## Endpoints

```
GET  /health
GET  /collections?refresh=true
GET  /tenants/{ai_tenant_id}
POST /tenants/{ai_tenant_id}/bootstrap
POST /search
POST /search/upload
POST /search/text
```

Search by URL:

```bash
curl -X POST https://your-service/search \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $IMAGE_SEARCH_API_TOKEN' \
  -d '{"ai_tenant_id":"tn_24_30","image_url":"https://example.com/dress.jpg","limit":5}'
```

Search by upload:

```bash
curl -X POST https://your-service/search/upload \
  -H 'Authorization: Bearer $IMAGE_SEARCH_API_TOKEN' \
  -F ai_tenant_id=tn_24_30 -F file=@dress.jpg
```

A result:

```json
{
  "ai_tenant_id": "tn_24_30",
  "collection": "tenant_products_tn_24_30_1787756447",
  "count": 1,
  "results": [{
    "sku": "CONFIGURABLE-7875",
    "product_id": 24083,
    "title": "فستان ردة مع بكلة كتف",
    "price": 249.0, "currency": "ILS", "in_stock": true,
    "image_url": "https://.../XbjbsKV.webp",
    "score": 0.9131,
    "colour": "أسود", "colour_in_stock": true, "colour_sizes": ["38","40","42"],
    "other_colours": [{"colour": "بني", "colour_sizes": ["40"], "score": 0.8802}]
  }]
}
```

Each picture of a product is indexed separately, so a good match returns the
same dress once per colour. `collapse_by_product` (on by default) merges them
into one row, keeps the colour whose picture scored highest, and lists the
rest under `other_colours` — because *which* picture matched is the
difference between "we have this dress" and "we have it in the colour you
photographed".

Bootstrap a tenant:

```bash
curl -X POST https://your-service/tenants/tn_99_1/bootstrap \
  -H 'Content-Type: application/json' -d '{}'
```

Comes back `skipped` if the tenant is already indexed (nothing embedded),
`accepted` if indexing started in the background, `locked` if another process
is already doing it. Poll `GET /tenants/{id}`. Pass `{"wait": true}` to hold
the request open instead — only sensible with a `limit`, since a full catalog
takes many minutes.

## Configuration

Copy `.env.example`. The two that matter:

- `QDRANT_URL` — **must** be the chatbot's Qdrant. Point it elsewhere and
  this service builds a second, parallel index nothing else can read.
- `DATABASE_URL` — the chatbot's Postgres, read-only. Only needed so a
  bootstrap can look up a tenant's Laravel credentials; searching works
  without it, and a bootstrap can carry `laravel_base_url` and
  `laravel_token` in the request instead.

`IMAGE_SEARCH_API_TOKEN` is a bearer token for every route except `/health`.
Empty means open — fine on a private network, wrong on a public URL.
`/health` reports which it is.

`ALLOW_FORCE_REINDEX` is off. Without it, `{"force": true}` is refused. The
ordinary way to rebuild a catalog is the chatbot's own sync, which builds
into a fresh collection and swaps only on success; this one writes in place.

## Running

```bash
docker build -t image-search .
docker run -p 8080:8080 --env-file .env image-search
```

The Dockerfile bakes the SigLIP weights into the image. Downloading them on
first request means the first customer to send a photograph waits for a 400MB
download, and a Hugging Face outage would stop a fresh deploy from serving at
all.

On Railway: `railway.toml` builds from the Dockerfile and health-checks
`/health` with a 300s window, which is what a cold container needs.

Gunicorn runs 2 workers, not the chatbot's 4 — each loads its own ~800MB copy
of the model, and this service is almost entirely one forward pass. Raise
`WEB_CONCURRENCY` on a bigger box.

## Tests

```bash
python tools/verify.py               # static: parses, no heavy top-level imports, payload contract
python tools/test_bootstrap_guard.py # the one rule, every path
python tools/test_results.py         # collapsing hits to products
python tools/test_mirror_in_step.py  # has the indexer drifted from the chatbot's?
python tools/check_live.py           # read-only: what each tenant would resolve to
python tools/check_live_search.py    # end-to-end against the real Qdrant
```

None of them import torch or transformers. `verify.py` is AST-only and never
imports the package at all, so the checks run on a machine with none of the
heavy stack installed.

`check_live_search.py` is the useful one. It reads a vector *back out* of a
tenant's collection and uses it as the query — which is exactly what the
model would produce for that product's own picture — so it exercises routing,
tenant resolution, the Qdrant search and the collapse to one row per product,
with the forward pass as the only thing stubbed. A correct run returns the
query vector's own product first at score 1.0.

## What it deliberately does not do

- **Own a vector store.** It reads and writes the chatbot's.
- **Write to the chatbot's Postgres.** The tenant row, the page row and the
  collection recorded on it belong to the chatbot; a bootstrap that repointed
  a page would take a live catalog away from it.
- **Keep a catalog current.** That is the chatbot's sync. This service
  indexes a tenant once, when there is nothing there, and then leaves it be.
