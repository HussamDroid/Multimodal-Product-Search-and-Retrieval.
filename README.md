# SouqLens

Multimodal product search for English, Arabic and reference images. The goal is to compare retrieval approaches with explicit relevance judgments, then ship a searchable catalog with stock and price filters.

## Checkpoint 1 — catalog and lexical baseline

- Validated CSV ingestion with stable product and variant-group IDs.
- English/Arabic tokenization and BM25 baseline.
- Unit tests for data validation and path boundaries.
- No real store data or credentials are included.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
souqlens baseline --catalog data/sample/catalog.csv --query "glass jar"
```

Next: aligned text/image encoders, persistent vector indexing, evaluation, reranking and a local search API. No model-quality or business-impact claims have been measured yet.
