"""Portable vector indexes with catalog and model provenance."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .catalog import Product, fingerprint, resolve_image
from .encoders import Encoder, normalize


class VectorIndex:
    def __init__(self, products, text_vectors, image_vectors, has_image, metadata):
        self.products = products
        self.text_vectors = normalize(text_vectors)
        self.image_vectors = normalize(image_vectors)
        self.has_image = np.asarray(has_image, dtype=bool)
        self.metadata = metadata
        shape = self.text_vectors.shape
        if (shape[0] != len(products) or shape[1] == 0 or self.image_vectors.shape != shape
                or self.has_image.shape != (len(products),)):
            raise ValueError("Index rows, dimensions and product IDs do not align")
        if len({p.product_id for p in products}) != len(products):
            raise ValueError("Index contains duplicate product IDs")
        if (np.linalg.norm(self.text_vectors, axis=1) == 0).any():
            raise ValueError("Product text embeddings cannot be zero")
        if (np.linalg.norm(self.image_vectors[self.has_image], axis=1) == 0).any():
            raise ValueError("Available image embeddings cannot be zero")

    @classmethod
    def build(cls, products: list[Product], image_root: Path, encoder: Encoder):
        text_vectors = normalize(encoder.text([p.text for p in products]))
        image_vectors = np.zeros_like(text_vectors)
        present = [i for i, p in enumerate(products) if p.image]
        paths = [resolve_image(image_root, products[i].image) for i in present]
        if paths:
            image_vectors[present] = normalize(encoder.images(paths))
        metadata = {"schema_version": 1, "built_at": datetime.now(timezone.utc).isoformat(),
                    "catalog_sha256": fingerprint(products), "encoder": encoder.identity,
                    "image_sha256": {products[i].product_id: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for i, path in zip(present, paths)}}
        return cls(products, text_vectors, image_vectors,
                   np.array([bool(p.image) for p in products]), metadata)

    def save(self, directory: Path):
        if directory.exists():
            raise ValueError("Index directory already exists; choose a new versioned output path")
        directory.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".souqlens-", dir=directory.parent))
        try:
            np.savez_compressed(temp / "vectors.npz", text=self.text_vectors,
                                image=self.image_vectors, has_image=self.has_image)
            manifest = {**self.metadata, "products": [asdict(p) for p in self.products],
                        "vectors_sha256": hashlib.sha256((temp / "vectors.npz").read_bytes()).hexdigest()}
            (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.rename(directory)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    @classmethod
    def load(cls, directory: Path):
        meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if meta.get("schema_version") != 1:
            raise ValueError("Unsupported index schema")
        if hashlib.sha256((directory / "vectors.npz").read_bytes()).hexdigest() != meta["vectors_sha256"]:
            raise ValueError("Vector checksum does not match the manifest")
        products = [Product(**p) for p in meta.pop("products")]
        if fingerprint(products) != meta["catalog_sha256"]:
            raise ValueError("Catalog checksum does not match the manifest")
        with np.load(directory / "vectors.npz", allow_pickle=False) as data:
            return cls(products, data["text"], data["image"], data["has_image"], meta)
