"""Retrieval metrics, judged query validation and slice-based experiment reports."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .catalog import fingerprint, resolve_image
from .search import Filters


def load_queries(path: Path, products):
    queries, ids, group_splits = [], set(), {}
    catalog = {p.product_id: p for p in products}
    for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        query = json.loads(text)
        qid, split = query.get("query_id"), query.get("split")
        if not qid or qid in ids or split not in {"train", "dev", "test"}:
            raise ValueError(f"Query line {line}: need unique ID and train/dev/test split")
        ids.add(qid)
        if not query.get("text", "").strip() and not query.get("image"):
            raise ValueError(f"Query {qid}: no query text or image")
        modality = "combined" if query.get("image") and query.get("text") else "image" if query.get("image") else "text"
        if query.get("modality") != modality or query.get("language") not in {"en", "ar", "none"}:
            raise ValueError(f"Query {qid}: incorrect modality or language tag")
        judgments = query.get("relevance", {})
        if not judgments or not any(v > 0 for v in judgments.values()):
            raise ValueError(f"Query {qid}: needs at least one relevant product")
        constraints = Filters(**query.get("filters", {}))
        for pid, grade in judgments.items():
            if pid not in catalog or type(grade) is not int or not 0 <= grade <= 3:
                raise ValueError(f"Query {qid}: unknown product or invalid relevance grade")
            if grade > 0:
                if not constraints.allows(catalog[pid]):
                    raise ValueError(f"Query {qid}: positive judgment contradicts query filters")
                group = catalog[pid].group_id
                if group in group_splits and group_splits[group] != split:
                    raise ValueError(f"Product-family leakage: {group} appears in multiple splits")
                group_splits[group] = split
        queries.append(query)
    if not queries:
        raise ValueError("Query file is empty")
    return queries


def metrics(predicted, relevance, k):
    if k < 1:
        raise ValueError("Metric cutoff must be positive")
    ranked = list(dict.fromkeys(predicted))[:k]
    relevant = {pid for pid, grade in relevance.items() if grade > 0}
    if not relevant:
        raise ValueError("Cannot compute recall without relevant products")
    gains = [relevance.get(pid, 0) for pid in ranked]
    dcg = sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(ideal))
    return {f"recall@{k}": len(set(ranked) & relevant)/len(relevant),
            f"ndcg@{k}": dcg/idcg if idcg else 0.0,
            f"mrr@{k}": next((1/(i+1) for i, pid in enumerate(ranked) if pid in relevant), 0.0)}


def run_query(engine, query, image_root, mode, k, ranker=None):
    return engine.search(text=query.get("text", ""),
                         image=resolve_image(image_root, query["image"]) if query.get("image") else None,
                         mode=mode, top_k=k, candidate_k=max(50, k),
                         filters=Filters(**query.get("filters", {})), ranker=ranker)


def evaluate(engine, queries, *, image_root, mode="bm25", split="test", k=5, ranker=None):
    selected = [q for q in queries if q["split"] == split]
    supported = [q for q in selected if mode != "bm25" or q["modality"] == "text"]
    if not supported:
        raise ValueError("No supported queries in the selected split")
    if ranker:
        test_groups = {p.group_id for q in supported for p in engine.products
                       if q["relevance"].get(p.product_id, 0) > 0}
        if set(ranker.metadata["training_groups"]) & test_groups and split != "train":
            raise ValueError("Reranker training groups overlap evaluation groups")
    cold_start = time.perf_counter()
    run_query(engine, supported[0], image_root, mode, k, ranker)
    cold_start = (time.perf_counter()-cold_start)*1000
    rows, slices = [], defaultdict(list)
    for query in supported:
        response = run_query(engine, query, image_root, mode, k, ranker)
        predictions = [r["product"]["product_id"] for r in response["results"]]
        row = {"query_id": query["query_id"], "language": query["language"],
               "modality": query["modality"], "latency_ms": response["latency_ms"],
               "predictions": predictions, **metrics(predictions, query["relevance"], k)}
        rows.append(row)
        slices["overall"].append(row)
        slices[f"{query['language']}/{query['modality']}"].append(row)
    summary = {}
    for name, values in slices.items():
        summary[name] = {"queries": len(values), **{key: float(np.mean([v[key] for v in values]))
                        for key in (f"recall@{k}", f"ndcg@{k}", f"mrr@{k}")},
                        "latency_p50_ms": float(np.percentile([v["latency_ms"] for v in values], 50)),
                        "latency_p95_ms": float(np.percentile([v["latency_ms"] for v in values], 95))}
    return {"created_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
            "reranked": bool(ranker), "split": split, "cutoff": k,
            "catalog_sha256": fingerprint(engine.products),
            "queries_sha256": hashlib.sha256(json.dumps(queries, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "encoder": engine.index.metadata["encoder"] if engine.index and mode != "bm25" else None,
            "cold_start_first_query_ms": cold_start,
            "unsupported_queries": [q["query_id"] for q in selected if q not in supported],
            "summary": summary, "per_query": rows,
            "notes": ["Latency excludes an initial warm-up query and includes query encoding.",
                      "Unjudged results count as zero relevance; label the evaluation pool carefully.",
                      "Synthetic fixture results are pipeline checks, not real catalog accuracy."]}
