# Verification record

## Software checks

- 22 unit/integration tests passed on Python 3.12.
- BM25 CLI search returned the jar variants for both English and Arabic queries.
- BM25 evaluation completed on the synthetic test split: four supported text queries, with the two image queries reported as unsupported.
- API tests exercised text search, image+text requests, filters, invalid uploads, upload size limits, invalid ranges and missing reranker handling.
- Dense retrieval and reranker tests use deterministic test encoders. Those results do not establish the quality of the pretrained models.

## Model execution

Both pretrained encoders loaded successfully on CPU and built an index for the 12-product synthetic catalog. The index records these resolved model revisions:

- Multilingual text: `58edf8cada9e398793dca955574a48cbb7f18be2`
- CLIP image: `327ab6726d33c0e22f920c83f2ff9e4bd38ca37f`

Dense and hybrid retrieval completed on six synthetic test queries: two English, two Arabic and two image-only. BM25 completed on the four supported text queries. A pairwise reranker trained on 120 explicit preference pairs from six train-split queries; held-out evaluation then completed with the trained model. The combined reference-photo/text query ranked `jar-1` and `jar-2` first and second.

The measured fixture results, model revisions and training query IDs are recorded in [fixture-smoke-results.json](fixture-smoke-results.json). These are synthetic smoke checks, not estimates of real-catalog performance. The tiny fixture does not show that reranking improves search; several methods already saturate its metrics.

The real encoders were also exercised through the API: English and Arabic hybrid queries with reranking, a combined image/text request, product-image serving and the model-loaded health state. All passed.

The demo JavaScript passed a syntax check. Visual browser QA and a Docker image build have not been performed. GitHub Actions completed successfully for checkpoints 1 through 4; CI intentionally uses download-free test encoders.

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
