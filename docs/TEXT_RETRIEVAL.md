# Dedicated multilingual text retrieval

The original dense path compares a CLIP query against both text and image vectors. That remains the version 1 baseline. Version 2 optionally adds an E5 text model (`intfloat/multilingual-e5-small` or `intfloat/multilingual-e5-base`) for text retrieval, while retaining the original CLIP vectors for reference photos.

E5-small uses 384-dimensional embeddings and E5-base uses 768-dimensional embeddings; CLIP uses its own aligned image/text space. The index stores these arrays separately. Text-only requests use E5. Photo-only requests retain the CLIP calculation. Combined requests use a fixed weighted average of scalar text and visual similarities, with the existing image weight default of 0.65. Those combined scores are not calibrated probabilities, and the two towers' scores are not independently calibrated.

E5 inputs use `query: ` and `passage: ` prefixes, including for Arabic. Product passages use the existing full product text. There is no query translation, generated Arabic product metadata, encoder fine-tuning or real-catalog reranker in this experiment.

Model revisions, passage recipe and dependency version are recorded in encoder provenance. Version 2 indexes require a valid semantic array; old version 1 bundles continue to load. A reranker trained on the old encoder is incompatible with the new index and must be retrained on appropriate judgments.

## Reproduce locally

Install the ML extra and build a new index from the same reviewed catalog:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml,import]"
.\.venv\Scripts\souqlens.exe index --catalog data/private/blackrose-v1/catalog.csv --image-root data/private/blackrose-v1/images --semantic-text --semantic-model intfloat/multilingual-e5-base --output artifacts/blackrose-index-v3
.\.venv\Scripts\souqlens.exe compare --index artifacts/blackrose-index-v1 --index artifacts/blackrose-index-v3 --queries data/private/blackrose-v1/queries-v1.jsonl --image-root data/private/blackrose-v1/images --output artifacts/blackrose-comparison-v1.json
```

Use `--semantic-revision`, `--text-revision` and `--image-revision` with the recorded full commit hashes to reproduce an exact encoder configuration. Use fresh output paths for later experiments. `compare` accepts text-only query sets so BM25, dense and hybrid receive the same query IDs. It rejects catalog changes between indexes and reports draft/reviewed/unspecified judgment counts, Hit@1, Recall@K, nDCG@K, MRR@K and warm CPU latencies.

To deploy the new index, export a new bundle and upload its contents to the private catalog dataset. The Space automatically selects the encoder configuration stored in the uploaded index. Point `SOUQLENS_DATASET_REVISION` to that dataset commit and restart the Space. Deploying application source alone does not activate a new data index.

## Black Rose draft evaluation

The owner supplied 18 products and verified text, indexed-photo and a new-photo search in the live app. The new photo itself was not supplied for offline evaluation, so no held-out image benchmark is claimed.

The text regression set contains 32 assistant-authored queries: 12 development queries and 20 test queries, evenly divided between English and Arabic. Fans, carpets and choppers are development intents; graters, trash bins, counters, cutting boards and serving trays are test intents. Positive product families do not cross splits. Every catalog product has a draft grade for every query, based on product intent. These are provisional judgments, not independent customer annotations or exhaustive relevance truth. Cross-category uses may require revised grades after owner review.

The full-product-text recipe was retained after a shorter title/category passage worsened development results. E5-small and E5-base were compared on development queries. E5-base placed five of six Arabic development queries first, versus four for E5-small; their development Arabic MRR@3 was the same. Base was chosen for its higher first-place count, at the cost of a larger model. The original test results had been inspected before that second model experiment, so the final test scores are exploratory regression results, not an untouched holdout. Further model selection needs a fresh, independently authored evaluation set. The earlier desk-fan and cutting-board smoke checks had already exposed some catalog intents, so this is a family-separated regression comparison, not a pristine unseen benchmark. Do not quote the draft results as validated store search accuracy.

Review `queries-v1.jsonl` and the evaluation report in the private bundle. Correct relevance grades before setting `judgment_status` to `reviewed`. Keep file versions and hashes: changing labels changes the experiment. Future work needs independently authored queries, no-match cases, more product families and owner-reviewed new reference photos.

References:
- https://huggingface.co/intfloat/multilingual-e5-small
- https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1

- https://huggingface.co/intfloat/multilingual-e5-base

## Recorded draft results

These are exploratory scores against assistant-authored labels, with owner review pending. The test-labelled portion contains 10 English and 10 Arabic text queries across six product families. The catalog has 18 products. Model choice was revisited after the first test report; this is not independent validation.

| Retrieval path | English Hit@1 | Arabic Hit@1 | Arabic Recall@3 | Arabic MRR@3 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 10/10 | 0/10 | 0.000 | 0.000 |
| Original CLIP dense | 5/10 | 4/10 | 0.700 | 0.567 |
| Original CLIP hybrid | 9/10 | 4/10 | 0.700 | 0.567 |
| E5-small dense / hybrid | 10/10 | 4/10 | 0.750 | 0.600 |
| E5-base dense / hybrid | 10/10 | 6/10 | 1.000 | 0.767 |

Dense and hybrid tie on these Arabic queries because the English-only catalog has no lexical Arabic matches. This experiment does not establish that hybrid always improves on dense retrieval. BM25 already retrieves the English query targets first on this small set.

The E5-base candidate improves the draft Arabic ranking, but four Arabic queries still do not put a relevant item first. Warm hybrid median CPU search time for the 20 test-labelled queries rose from about 20.1 ms to 35.8 ms with four OpenMP threads. These local measurements include query encoding and exclude warm-up and hosted queue time; they are not a controlled hardware benchmark or a guarantee of Space latency.

The full JSON reports, per-query predictions and draft labels accompany the private deployment bundle. Photos and product metadata are unchanged; the candidate bundle adds a separately versioned semantic text array. It contains no synthetic reranker. A catalog-cover photo and combined-query smoke check must pass before handoff, while fresh user photos remain a manual deployment check.
