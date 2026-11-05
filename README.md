# SouqLens

Multimodal product search for English, Arabic and reference images. The goal is to compare retrieval approaches with explicit relevance judgments, then ship a searchable catalog with stock and price filters.

## Checkpoint 3 — evaluation and supervised reranking

- Validated CSV ingestion with stable product and variant-group IDs.
- English/Arabic tokenization and BM25 baseline.
- Unit tests for data validation and path boundaries.
- Lazy-loaded multilingual CLIP text encoder aligned to a CLIP image encoder.
- Text, image, combined-query, dense and RRF hybrid retrieval.
- Stock/category/price filters applied before ranking.
- Checksummed indexes with catalog, model and image provenance.
- Recall@K, nDCG@K and MRR@K with English/Arabic/image slices and warm latency percentiles.
- Validation that relevant product families do not cross train/dev/test query splits.
- Pairwise linear reranker trained only on explicit train-split judgments.
- No real store data or credentials are included.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
souqlens baseline --catalog data/sample/catalog.csv --query "glass jar"
```

```bash
python -m pip install -e '.[ml]'
souqlens index --catalog data/sample/catalog.csv --image-root data/sample --output artifacts/index-v1
souqlens search --index artifacts/index-v1 --query "برطمان زجاجي" --in-stock
souqlens search --index artifacts/index-v1 --image data/sample/images/jar-query.jpg --query "small jar" --max-price 30
```

```bash
souqlens evaluate --catalog data/sample/catalog.csv --queries data/sample/queries.jsonl --image-root data/sample --mode bm25 --split test --output artifacts/bm25.json
souqlens evaluate --index artifacts/index-v1 --queries data/sample/queries.jsonl --image-root data/sample --mode hybrid --split test --output artifacts/hybrid.json
souqlens train-ranker --index artifacts/index-v1 --queries data/sample/queries.jsonl --image-root data/sample --output artifacts/ranker-v1.json
souqlens evaluate --index artifacts/index-v1 --queries data/sample/queries.jsonl --image-root data/sample --mode hybrid --ranker artifacts/ranker-v1.json --split test --output artifacts/reranked.json
```

The committed catalog, queries and illustrations are synthetic fixtures. They check the experiment pipeline and do not establish real-world retrieval accuracy. BM25 skips image/combined queries explicitly instead of presenting them as supported.

Neural model execution requires downloading pretrained weights. The deterministic encoder doubles in unit tests only verify retrieval plumbing, not real embedding quality. Next: the local API/demo and real-catalog evaluation. No business-impact claims have been measured.
