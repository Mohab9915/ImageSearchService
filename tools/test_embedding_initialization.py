"""Concurrent requests share one complete model and processor."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Barrier
from time import sleep
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagesearch import embedding


if __name__ == "__main__":
    model, processor = Mock(), Mock()
    start = Barrier(8)

    def load_processor(*_):
        sleep(0.05)
        return processor

    def request(_):
        start.wait(timeout=5)
        return embedding.get_embedding_model()

    with (
        patch("transformers.SiglipModel.from_pretrained", return_value=model) as load_model,
        patch("transformers.SiglipProcessor.from_pretrained", side_effect=load_processor) as load,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        assert list(pool.map(request, range(8))) == [(model, processor)] * 8
        load_model.assert_called_once_with(embedding.SIGLIP_MODEL)
        load.assert_called_once_with(embedding.SIGLIP_MODEL)
