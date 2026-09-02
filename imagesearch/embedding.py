"""SigLIP embeddings.

MIRROR. This is a copy of the embedding half of the chatbot's
`helpers/webhooks/qdrant_image_search.py`. It is duplicated rather than
imported because the two services deploy independently -- but the duplication
is a contract, not a convenience. The checkpoint, the pooling fallback, the
L2 normalisation and the 768 dimensions must stay identical in both, or the
vectors this service writes and the vectors the chatbot searches stop living
in the same space. Nothing raises when that happens; search simply returns
the wrong products.
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

import httpx

from imagesearch.config import SIGLIP_MODEL

logger = logging.getLogger(__name__)

_model = None
_processor = None


def get_embedding_model():
    """The SigLIP model and processor, loaded once per process."""
    global _model, _processor

    if _model is None:
        try:
            # transformers has shipped releases where siglip has no tokenizer
            # registered, which makes SiglipProcessor.from_pretrained fail on
            # a model that is otherwise fine.
            try:
                from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
                if TOKENIZER_MAPPING_NAMES.get("siglip") is None:
                    TOKENIZER_MAPPING_NAMES["siglip"] = "SiglipTokenizer"
            except Exception as pe:
                logger.warning("[SigLIP] Could not patch transformers: %s", pe)

            from transformers import SiglipModel, SiglipProcessor

            logger.info("[SigLIP] Loading %s ...", SIGLIP_MODEL)
            _model = SiglipModel.from_pretrained(SIGLIP_MODEL)
            _processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL)
            _model.eval()
            logger.info("[SigLIP] Model loaded")
        except Exception as e:
            logger.error("[SigLIP] Failed to load model: %s", e)
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    return _model, _processor


def warmup_model() -> None:
    """Run one forward pass at startup.

    The first inference in a process pays for lazy weight materialisation and
    kernel selection. Paying it here means the first customer photograph does
    not.
    """
    try:
        model, processor = get_embedding_model()
        if model is None or processor is None:
            return

        import numpy as np
        import torch
        from PIL import Image

        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        inputs = processor(images=dummy, return_tensors="pt")
        with torch.no_grad():
            model.get_image_features(**inputs)
        logger.info("[SigLIP] Warmed up")
    except Exception as e:
        logger.warning("[SigLIP] Warmup failed: %s", e)


def _normalise(features):
    """Unit-length vector as a list of floats.

    Some transformers versions return a ModelOutput rather than a tensor, so
    reach for the pooled representation before falling back to the first
    token of the last hidden state.
    """
    import torch
    import torch.nn.functional as F

    if not isinstance(features, torch.Tensor):
        if getattr(features, "pooler_output", None) is not None:
            features = features.pooler_output
        elif hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state[:, 0]

    return F.normalize(features, p=2, dim=-1).squeeze().tolist()


def embed_image_bytes(data: bytes) -> Optional[list]:
    """Embed an image already in memory."""
    import torch
    from PIL import Image

    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        logger.warning("[SigLIP] Cannot decode image bytes: %s", e)
        return None

    model, processor = get_embedding_model()
    if model is None or processor is None:
        return None

    try:
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            return _normalise(model.get_image_features(**inputs))
    except Exception as e:
        logger.error("[SigLIP] Image embedding failed: %s", e)
        return None


def embed_image_url(image_url: str, timeout: float = 15.0) -> Optional[list]:
    """Fetch an image and embed it.

    A URL that answers with HTML -- an expired CDN link, a login wall, a 404
    page served with status 200 -- is not an image, and PIL would either
    raise or, worse, decode something meaningless. Check before decoding.
    """
    if not image_url:
        return None
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (ImageSearch/1.0)"},
        ) as client:
            resp = client.get(image_url)

        if resp.status_code >= 400:
            logger.warning("[SigLIP] Image fetch %s: HTTP %s", image_url[:80], resp.status_code)
            return None

        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            logger.warning("[SigLIP] %s is not an image (content-type %s)",
                           image_url[:80], content_type or "missing")
            return None

        return embed_image_bytes(resp.content)
    except Exception as e:
        logger.error("[SigLIP] Image embedding failed for %s: %s", image_url[:80], e)
        return None


def embed_text(text: str) -> Optional[list]:
    """Embed a phrase into the same space as the pictures."""
    if not text or not text.strip():
        return None

    import torch

    model, processor = get_embedding_model()
    if model is None or processor is None:
        return None

    try:
        with torch.no_grad():
            inputs = processor(
                text=[text], return_tensors="pt",
                padding="max_length", truncation=True, max_length=64,
            )
            return _normalise(model.get_text_features(**inputs))
    except Exception as e:
        logger.error("[SigLIP] Text embedding failed: %s", e)
        return None
