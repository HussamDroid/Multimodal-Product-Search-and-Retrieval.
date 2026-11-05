"""Lexical, dense and hybrid retrieval with deterministic ranking and filters."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import numpy as np

from .catalog import fingerprint
from .encoders import normalize
from .lexical import BM25, tokenize


@dataclass(frozen=True)
class Filters:
    min_price: float | None = None
    max_price: float | None = None
    in_stock: bool = False
    category: str | None = None

    def __post_init__(self):
        for value in (self.min_price, self.max_price):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Price bounds must be finite and nonnegative")
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("Minimum price exceeds maximum price")

    def allows(self, product):
        return ((self.min_price is None or product.price >= self.min_price)
                and (self.max_price is None or product.price <= self.max_price)
                and (not self.in_stock or product.stock > 0)
                and (not self.category or product.category.casefold() == self.category.casefold()))


class SearchEngine:
    def __init__(self, products, index=None, encoder=None, encoder_factory=None):
        self.products, self.index = products, index
        self.encoder, self.encoder_factory = encoder, encoder_factory
        self.lexical = BM25([p.text for p in products])
        if index and fingerprint(products) != index.metadata["catalog_sha256"]:
            raise ValueError("The catalog does not match the vector index; rebuild the index")

    def _encoder(self):
        if self.encoder is None:
            if self.encoder_factory is None:
                raise ValueError("No embedding encoder configured")
            self.encoder = self.encoder_factory()
        if self.encoder.identity != self.index.metadata["encoder"]:
            raise ValueError("Query encoder differs from the index encoder; rebuild or use matching revisions")
        return self.encoder

    def search(self, text="", image: Path | None = None, mode="hybrid", top_k=10,
               candidate_k=50, filters: Filters | None = None, image_weight=0.65, ranker=None):
        started = time.perf_counter()
        text = text.strip()
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Mode must be bm25, dense or hybrid")
        if not text and image is None:
            raise ValueError("Provide text, an image, or both")
        if not 1 <= top_k <= 100 or not top_k <= candidate_k <= 1000:
            raise ValueError("Require 1 <= top_k <= 100 and top_k <= candidate_k <= 1000")
        if not 0 <= image_weight <= 1:
            raise ValueError("Image weight must be between zero and one")
        if image is not None and mode == "bm25":
            raise ValueError("BM25 does not support image queries")
        if ranker is not None and mode != "hybrid":
            raise ValueError("The learned reranker requires hybrid retrieval")
        if mode != "bm25" and self.index is None:
            raise ValueError("Build and configure a dense index before using semantic search")
        eligible = [i for i, p in enumerate(self.products) if (filters or Filters()).allows(p)]
        lexical = self.lexical.scores(text)
        dense = np.zeros(len(self.products), dtype=np.float32)
        if mode != "bm25":
            encoder = self._encoder()
            vector = None
            if text:
                vector = encoder.text([text])
            if image is not None:
                picture = encoder.images([image])
                vector = picture if vector is None else image_weight * picture + (1-image_weight) * vector
            vector = normalize(vector)
            if vector.shape != (1, self.index.text_vectors.shape[1]) or np.linalg.norm(vector) == 0:
                raise ValueError("Query embedding has an invalid shape or zero norm")
            text_score = self.index.text_vectors @ vector[0]
            image_score = self.index.image_vectors @ vector[0]
            dense = np.where(self.index.has_image, (text_score + image_score) / 2, text_score)
        order = lambda scores, positive=False: sorted(
            (i for i in eligible if not positive or scores[i] > 0),
            key=lambda i: (-float(scores[i]), self.products[i].product_id))[:candidate_k]
        lexical_order, dense_order = order(lexical, True), order(dense)
        scores = {}
        if mode == "bm25":
            scores = {i: float(lexical[i]) for i in lexical_order}
        elif mode == "dense":
            scores = {i: float(dense[i]) for i in dense_order}
        else:
            # RRF avoids pretending raw BM25 scores and cosine similarities share a scale.
            for ranking in (lexical_order, dense_order):
                for rank, i in enumerate(ranking, 1):
                    scores[i] = scores.get(i, 0.0) + 1 / (60 + rank)
        candidates = sorted(scores, key=lambda i: (-scores[i], self.products[i].product_id))
        if ranker is not None and candidates:
            values = ranker.predict(np.array([self.features(text, i, lexical[i], dense[i]) for i in candidates]))
            scores = dict(zip(candidates, map(float, values)))
            candidates.sort(key=lambda i: (-scores[i], self.products[i].product_id))
        results = [{"product": asdict(self.products[i]), "score": scores[i],
                    "bm25": float(lexical[i]), "cosine": float(dense[i]) if mode != "bm25" else None,
                    "features": self.features(text, i, lexical[i], dense[i])}
                   for i in candidates[:top_k]]
        return {"mode": mode, "reranked": ranker is not None, "results": results,
                "eligible_products": len(eligible), "latency_ms": (time.perf_counter()-started)*1000,
                "score_note": "Ranking scores are not calibrated probabilities."}

    def features(self, text, i, lexical, dense):
        query = set(tokenize(text))
        name = set(tokenize(self.products[i].name_en + " " + self.products[i].name_ar))
        return [float(lexical / (lexical + 1)), float(dense), len(query & name) / max(1, len(query))]
