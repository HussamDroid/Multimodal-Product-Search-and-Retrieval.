"""Validated product records and portable catalog loading."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Product:
    product_id: str
    group_id: str
    barcode: str
    name_en: str
    name_ar: str
    description_en: str
    description_ar: str
    category: str
    price: float
    stock: int
    image: str = ""

    @property
    def text(self) -> str:
        return " . ".join(filter(None, [self.name_en, self.name_ar,
                                      self.description_en, self.description_ar, self.category]))


def resolve_image(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image paths must stay inside the configured image root.")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError(f"Unsupported image format: {relative}")
    return path


def load_catalog(path: Path, image_root: Path | None = None) -> list[Product]:
    products, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"product_id", "group_id", "name_en", "price", "stock"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV needs columns: {', '.join(sorted(required))}")
        for line, row in enumerate(reader, 2):
            if not any(row.values()):
                continue
            try:
                if None in row:
                    raise ValueError("More values than column headings")
                values = {key: (row.get(key) or "").strip() for key in Product.__dataclass_fields__}
                if not all(values[key] for key in ("product_id", "group_id", "name_en")):
                    raise ValueError("Product ID, group ID and English name cannot be blank")
                if values["product_id"] in seen:
                    raise ValueError("Duplicate product ID")
                price, stock = float(values["price"]), int(values["stock"])
                if not math.isfinite(price) or price < 0 or stock < 0:
                    raise ValueError("Price must be finite and stock must be nonnegative")
                values.update(price=price, stock=stock)
                if image_root and values["image"]:
                    if not resolve_image(image_root, values["image"]).is_file():
                        raise ValueError(f"Image not found: {values['image']}")
                products.append(Product(**values))
                seen.add(values["product_id"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Catalog line {line}: {error}") from error
    if not products:
        raise ValueError("Catalog is empty")
    return products


def fingerprint(products: list[Product]) -> str:
    payload = json.dumps([asdict(p) for p in products], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
