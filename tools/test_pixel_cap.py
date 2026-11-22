"""An image that decodes to hundreds of MB is refused before decoding."""
import io
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.modules.setdefault("torch", types.ModuleType("torch"))  # not needed: refused before inference

from PIL import Image  # noqa: E402

from imagesearch import embedding  # noqa: E402

embedding.get_embedding_model = lambda: (_ for _ in ()).throw(AssertionError("model must not load"))
b = io.BytesIO()
Image.new("L", (13000, 13000), 0).save(b, "PNG")
assert embedding.embed_image_bytes(b.getvalue()) is None
print("PASS oversized-pixel image refused")
