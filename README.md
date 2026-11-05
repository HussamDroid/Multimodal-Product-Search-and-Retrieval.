# SouqLens

Multimodal product search for English, Arabic and reference images. The goal is to compare retrieval approaches with explicit relevance judgments, then ship a searchable catalog with stock and price filters.

## Checkpoint 2 — multimodal retrieval

- Validated CSV ingestion with stable product and variant-group IDs.
- English/Arabic tokenization and BM25 baseline.
- Unit tests for data validation and path boundaries.
- Lazy-loaded multilingual CLIP text encoder aligned to a CLIP image encoder.
- Text, image, combined-query, dense and RRF hybrid retrieval.
- Stock/category/price filters applied before ranking.
- Checksummed indexes with catalog, model and image provenance.
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

Neural model execution requires downloading pretrained weights. The deterministic encoder doubles in unit tests only verify retrieval plumbing, not real embedding quality. Next: evaluation, supervised reranking and a local search API. No model-quality or business-impact claims have been measured yet.
