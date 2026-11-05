# Verification record

## Software checks

- 22 unit/integration tests passed on Python 3.12.
- BM25 CLI search returned the jar variants for both English and Arabic queries.
- BM25 evaluation completed on the synthetic test split: four supported text queries, with the two image queries reported as unsupported.
- API tests exercised text search, image+text requests, filters, invalid uploads, upload size limits, invalid ranges and missing reranker handling.
- Dense retrieval and reranker tests use deterministic test encoders. Those results do not establish the quality of the pretrained models.

## Model execution

Pretrained model download and inference verification is in progress. A successful model run must be recorded separately before describing semantic retrieval as execution-verified.

## Development environment

| Package | Version observed |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| Pillow | 13.0.0 |
| FastAPI | 0.162.0 |
| Uvicorn | 0.54.0 |
| python-multipart | 0.0.33 |
| httpx | 0.28.1 |
| sentence-transformers | 5.7.0 |
| transformers | 5.22.1 |
| torch | 2.15.0+cpu |
| huggingface-hub | 2.4.0 |

These are observed versions, not a promise that the latest releases are equivalent. Record exact dependency versions and model revision hashes with each real experiment. No GPU throughput, concurrency benchmark, real-catalog accuracy, production availability or business uplift has been measured.
