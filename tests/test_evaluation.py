import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from souqlens.evaluation import evaluate, load_queries, metrics
from souqlens.ranker import PairwiseRanker
from test_retrieval import fixture


class EvaluationTests(unittest.TestCase):
    def test_metric_values_and_deduplication(self):
        actual = metrics(["a", "a", "b"], {"a": 2, "b": 1}, 2)
        self.assertEqual(actual["recall@2"], 1)
        self.assertEqual(actual["ndcg@2"], 1)
        self.assertEqual(actual["mrr@2"], 1)
        self.assertEqual(metrics(["wrong", "a"], {"a": 1}, 2)["mrr@2"], .5)
        self.assertEqual(metrics([], {"a": 1}, 2)["recall@2"], 0)

    def test_group_leakage_rejected(self):
        qs = [{"query_id": "one", "text": "jar", "language": "en", "modality": "text",
               "split": "train", "relevance": {"jar": 2}},
              {"query_id": "two", "text": "برطمان", "language": "ar", "modality": "text",
               "split": "test", "relevance": {"jar": 2}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"queries.jsonl"
            path.write_text("\n".join(json.dumps(q) for q in qs))
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_queries(path, fixture().products)

    def test_bm25_reports_unsupported_image_queries(self):
        queries = [{"query_id": "text", "text": "jar", "language": "en", "modality": "text",
                    "split": "test", "relevance": {"jar": 2}},
                   {"query_id": "image", "image": "fixture.jpg", "language": "none", "modality": "image",
                    "split": "test", "relevance": {"jar": 2}}]
        report = evaluate(fixture(), queries, image_root=Path("."), mode="bm25", k=1)
        self.assertEqual(report["unsupported_queries"], ["image"])
        self.assertEqual(report["summary"]["overall"]["queries"], 1)
        self.assertEqual(report["summary"]["en/text"]["recall@1"], 1)

    def test_training_excludes_dev_and_test_and_learns_preferences(self):
        train = {"query_id": "train-jar", "text": "jar", "split": "train",
                 "relevance": {"jar": 2, "lamp": 0}}
        heldout = {"query_id": "test-lamp", "text": "lamp", "split": "test",
                   "relevance": {"lamp": 2, "jar": 0}}
        engine = fixture()
        ranker = PairwiseRanker.train(engine, [train, heldout], Path("."))
        self.assertEqual(ranker.metadata["training_query_ids"], ["train-jar"])
        self.assertLess(ranker.metadata["loss_history"][-1]["pairwise_loss"],
                        ranker.metadata["loss_history"][0]["pairwise_loss"])
        self.assertGreater(ranker.predict([[1, 1, 1]])[0], ranker.predict([[0, 0, 0]])[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"ranker.json"
            ranker.save(path)
            np.testing.assert_allclose(PairwiseRanker.load(path, engine.index).weights, ranker.weights)

    def test_unjudged_products_do_not_become_negative_training_labels(self):
        with self.assertRaisesRegex(ValueError, "No judged training pairs"):
            PairwiseRanker.train(fixture(), [{"query_id": "x", "text": "jar", "split": "train",
                                            "relevance": {"jar": 2}}], Path("."))


if __name__ == "__main__":
    unittest.main()
