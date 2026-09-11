"""Every environment variable this service reads, in one place.

The values that matter are the ones shared with the chatbot: QDRANT_URL and
DATABASE_URL must name the *same* Qdrant and the *same* Postgres, or this
service will build a second, parallel index that nothing else can read.
"""
from __future__ import annotations

import os


def _flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- shared
# The same vector store the chatbot reads. Not a new one -- collections are
# looked up by the chatbot's own naming convention and are written in the
# chatbot's own payload shape.
QDRANT_URL = os.environ.get(
    "QDRANT_URL", "https://qdrant-production-efa1.up.railway.app"
).rstrip("/")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# The chatbot's Postgres, read-only from here. Used only to turn an
# ai_tenant_id into that tenant's Laravel base url and token so a bootstrap
# can pull the catalog. Leave unset and every request must carry them.
DATABASE_URL = os.environ.get("DATABASE_URL", "") or os.environ.get("NEW_DATABASE_URL", "")

# ---------------------------------------------------------------- model
# Must match the chatbot exactly. A different checkpoint produces vectors in
# a different space: search would return nonsense against collections the
# chatbot built, with no error to say why.
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL", "google/siglip-base-patch16-224")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "768"))

# ---------------------------------------------------------------- service
API_TOKEN = os.environ.get("IMAGE_SEARCH_API_TOKEN", "")
PORT = int(os.environ.get("PORT", "8080"))

# Search defaults. 0.68 is the chatbot's threshold for "this is the product
# in the photograph" as opposed to "this looks a bit like it".
DEFAULT_LIMIT = int(os.environ.get("SEARCH_LIMIT", "6"))
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("SEARCH_SCORE_THRESHOLD", "0.68"))

# Largest upload accepted on /search, in bytes.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))

# Requests per minute, per bearer token (or client IP when unauthenticated).
# 0 disables the limit. /health is never limited -- it's what Railway polls.
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))

# ---------------------------------------------------------------- bootstrap
# A bootstrap indexes a tenant that has no collection yet. It never touches a
# tenant that already has one -- that catalog belongs to the chatbot's sync
# and re-embedding it would burn hours of GPU-less CPU to produce the vectors
# that are already there.
#
# Set true only to deliberately rebuild, and only when nothing is reading.
ALLOW_FORCE_REINDEX = _flag("ALLOW_FORCE_REINDEX")

# How long a bootstrap lock stays valid before another process may assume the
# holder died. A 30k-product catalog takes tens of minutes.
BOOTSTRAP_LOCK_TTL = int(os.environ.get("BOOTSTRAP_LOCK_TTL", str(6 * 3600)))
LOCK_COLLECTION = os.environ.get("LOCK_COLLECTION", "imagesearch_bootstrap_locks")
