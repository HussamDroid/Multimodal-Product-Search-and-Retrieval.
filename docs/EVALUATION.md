# Evaluation protocol

The sample fixtures exist to make commands and tests reproducible. They are deliberately easy, use synthetic illustrations and repeat category phrases. A perfect fixture score is not evidence that search works on a real store.

## Build a useful judgment set

1. Export a permitted product catalog. Keep stable product IDs and group variants of the same item under a shared `group_id`.
2. Write realistic English and Arabic queries: paraphrases, misspellings, vague requests, exact attributes and queries with no suitable product. Keep query authors independent of product descriptions where possible.
3. Add separately captured reference photos for image queries. Avoid using the exact catalog image as a supposed unseen query.
4. Pool the top candidates from BM25, dense and hybrid retrieval. Judge the pooled results: 0 irrelevant, 1 loosely relevant, 2 useful match, 3 exact match. A second reviewer should audit a subset.
5. Assign relevant product families to train/dev/test, then put related queries in the same split. The loader rejects family overlap. It cannot detect mislabeled variant groups or copied photos automatically.
6. Keep test judgments fixed. Tune fusion weights, thresholds and ranker settings on train/dev only.

Query records are JSON Lines:

```json
{"query_id":"q001","text":"small glass jar","language":"en","modality":"text","split":"test","relevance":{"jar-1":3,"jar-2":1,"mug-1":0},"filters":{"max_price":40,"in_stock":true}}
```

Image queries use `image`, relative to `--image-root`; combined queries include both `image` and `text`. Use modality `image` or `combined` accordingly. Use language `none` for image-only queries. Positive judgments must satisfy explicit filters.

## Report comparisons fairly

- Compare BM25, dense, hybrid and hybrid + reranker on the same text-query subset.
- Report image-only and combined-query results separately. BM25 does not support those queries; the evaluator lists them as unsupported.
- Report Hit@1, macro Recall@K, nDCG@K, MRR@K and query counts per English/Arabic/modality slice.
- Latency includes query encoding and excludes one initial warm-up query. First-query latency is reported separately. Always record CPU/GPU, model revisions, corpus size and request concurrency alongside performance results.
- Scores are ranking functions, not purchase likelihoods or calibrated confidence.
- Unjudged predictions receive zero relevance in evaluation. Incomplete pools can bias comparisons, so expand judgments when a new method retrieves new candidates.
- The current report does not include confidence intervals, significance tests, dedicated no-match metrics or candidate recall. Add these before making strong comparative claims.

## Reranker scope

The pairwise linear model learns three feature weights: saturated BM25, cosine similarity and query-token coverage in product names. It only forms preferences between explicitly judged products with different grades. Unjudged products are not assumed negative during training.

This is supervised reranking, not fine-tuning the image/text encoders. The held-out evaluator rejects overlap with the ranker's recorded training families. The CLI validates all splits before training; direct library callers must use `load_queries` first.

Training retrieves up to 100 candidates per query. If a relevant product is absent, reranking cannot recover it. Use candidate recall to decide whether to improve the embedding model or the reranker next.

## Next model experiment

Once the baseline has a real judged dataset, mine difficult nonmatches among similar variants and train an embedding adapter or fine-tune with a contrastive objective. Keep all images and near-duplicate variants of each family in the same split. Record the unchanged pretrained baseline, trained configuration, seeds, dataset fingerprints and held-out metrics. Do not add encoder training merely to increase the project feature count.

For the optional E5 text tower and the draft Black Rose regression comparison, see [TEXT_RETRIEVAL.md](TEXT_RETRIEVAL.md). Query records may include `judgment_status` (`draft`, `reviewed` or `unspecified`); reports expose counts so provisional labels remain visible.
