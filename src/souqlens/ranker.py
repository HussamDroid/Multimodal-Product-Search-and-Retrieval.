"""An inspectable pairwise linear reranker trained on explicit relevance judgments."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluation import run_query

FEATURES = ["bm25_saturated", "cosine", "name_token_coverage"]


class PairwiseRanker:
    def __init__(self, weights, metadata):
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (len(FEATURES),) or not np.isfinite(self.weights).all():
            raise ValueError("Reranker weights have invalid shape or values")
        self.metadata = metadata

    def predict(self, features):
        return np.asarray(features) @ self.weights

    @classmethod
    def train(cls, engine, queries, image_root, steps=500, learning_rate=0.1, l2=0.01):
        if engine.index is None:
            raise ValueError("Build a dense index before training the hybrid reranker")
        pairs, used = [], []
        for query in queries:
            if query["split"] != "train":
                continue
            results = run_query(engine, query, image_root, "hybrid", min(100, len(engine.products)))["results"]
            # Unjudged products are not automatically negative training examples.
            judged = [r for r in results if r["product"]["product_id"] in query["relevance"]]
            count = len(pairs)
            for first in judged:
                for second in judged:
                    a, b = first["product"]["product_id"], second["product"]["product_id"]
                    if query["relevance"][a] > query["relevance"][b]:
                        pairs.append(np.array(first["features"]) - np.array(second["features"]))
            if len(pairs) > count:
                used.append(query)
        if not pairs:
            raise ValueError("No judged training pairs found. Label retrieved hard negatives with grade 0.")
        if steps < 1 or learning_rate <= 0 or l2 < 0:
            raise ValueError("Invalid optimizer settings")
        differences, weights = np.asarray(pairs), np.zeros(len(FEATURES))
        losses = []
        for step in range(steps):
            margins = differences @ weights
            gradient = -(differences / (1 + np.exp(np.clip(margins, -50, 50)))[:, None]).mean(axis=0) + l2*weights
            weights -= learning_rate * gradient
            if step % 50 == 0 or step == steps-1:
                losses.append({"step": step, "pairwise_loss": float(np.logaddexp(0, -(differences @ weights)).mean())})
        positives = {pid for q in used for pid, grade in q["relevance"].items() if grade > 0}
        metadata = {"schema_version": 1, "features": FEATURES, "training_pairs": len(pairs),
                    "training_query_ids": [q["query_id"] for q in used],
                    "training_groups": sorted({p.group_id for p in engine.products if p.product_id in positives}),
                    "catalog_sha256": engine.index.metadata["catalog_sha256"],
                    "encoder": engine.index.metadata["encoder"], "optimizer": {
                        "steps": steps, "learning_rate": learning_rate, "l2": l2},
                    "training_queries_sha256": hashlib.sha256(json.dumps(used, sort_keys=True).encode()).hexdigest(),
                    "loss_history": losses}
        return cls(weights, metadata)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            json.dump({"weights": self.weights.tolist(), "metadata": self.metadata}, file, indent=2)

    @classmethod
    def load(cls, path: Path, index):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["metadata"]
        if meta.get("schema_version") != 1 or meta.get("features") != FEATURES:
            raise ValueError("Incompatible reranker schema")
        if any(meta[key] != index.metadata[key] for key in ("encoder", "catalog_sha256")):
            raise ValueError("Reranker does not match this index")
        return cls(data["weights"], meta)
