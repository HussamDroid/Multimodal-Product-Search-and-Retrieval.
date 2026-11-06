"""Aligned multilingual text and CLIP image encoders, loaded only when needed."""

from importlib.metadata import version
from pathlib import Path
import re
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
SEMANTIC_MODEL = "intfloat/multilingual-e5-small"


class Encoder(Protocol):
    @property
    def identity(self) -> dict: ...
    def text(self, texts: list[str]) -> np.ndarray: ...
    def images(self, paths: list[Path]) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 0)


class ClipEncoder:
    def __init__(self, *, device="cpu", text_revision=None, image_revision=None, batch_size=16,
                 semantic_text=False, semantic_revision=None, semantic_model=SEMANTIC_MODEL):
        if semantic_text and semantic_model not in {SEMANTIC_MODEL, "intfloat/multilingual-e5-base"}:
            raise ValueError("Unsupported semantic text model")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install model dependencies first: pip install -e '.[ml]'") from error
        from huggingface_hub import HfApi
        def pinned(model, revision):
            if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
                return revision
            return HfApi().model_info(model, revision=revision).sha
        text_revision = pinned(TEXT_MODEL, text_revision)
        image_revision = pinned(IMAGE_MODEL, image_revision)
        self._identity = {"backend": "multilingual-clip", "text_model": TEXT_MODEL,
                          "image_model": IMAGE_MODEL, "text_revision": text_revision,
                          "image_revision": image_revision,
                          "sentence_transformers": version("sentence-transformers")}
        self.batch_size = batch_size
        self.text_model = SentenceTransformer(TEXT_MODEL, revision=text_revision, device=device,
                                               trust_remote_code=False)
        self.image_model = SentenceTransformer(IMAGE_MODEL, revision=image_revision, device=device,
                                                trust_remote_code=False)

        self.semantic_model = None
        if semantic_text:
            semantic_revision = pinned(semantic_model, semantic_revision)
            self._identity.update(semantic_model=semantic_model, semantic_revision=semantic_revision,
                                  semantic_recipe="full-product-text-v1")
            self.semantic_model = SentenceTransformer(semantic_model, revision=semantic_revision,
                                                      device=device, trust_remote_code=False)

    @property
    def identity(self):
        return dict(self._identity)

    def semantic_queries(self, texts):
        return self._semantic_encode(texts, "query: ")

    def semantic_documents(self, texts):
        return self._semantic_encode(texts, "passage: ")

    def _semantic_encode(self, texts, prefix):
        if self.semantic_model is None:
            raise ValueError("No dedicated text encoder configured")
        return normalize(self.semantic_model.encode([prefix + text for text in texts],
            batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False))

    def text(self, texts):
        return normalize(self.text_model.encode(texts, batch_size=self.batch_size,
                                               convert_to_numpy=True, show_progress_bar=False))

    def images(self, paths):
        vectors = []
        # Open one batch at a time so a large catalog does not keep every image in RAM.
        for start in range(0, len(paths), self.batch_size):
            images = []
            try:
                for path in paths[start:start+self.batch_size]:
                    with Image.open(path) as image:
                        image.load()
                        images.append(ImageOps.exif_transpose(image).convert("RGB"))
                vectors.append(self.image_model.encode(images, convert_to_numpy=True,
                                                        show_progress_bar=False))
            finally:
                for image in images:
                    image.close()
        if not vectors:
            raise ValueError("No images to encode")
        return normalize(np.concatenate(vectors, axis=0))
