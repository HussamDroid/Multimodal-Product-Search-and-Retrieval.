import tempfile
import unittest
from pathlib import Path

import numpy as np

from souqlens.catalog import Product, fingerprint
from souqlens.encoders import normalize
from souqlens.index import VectorIndex
from souqlens.search import Filters, SearchEngine


def fixture():
    products = [Product("jar", "jars", "001", "glass jar", "برطمان", "", "", "Kitchen", 20, 5),
                Product("lamp", "lamps", "002", "table lamp", "مصباح", "", "", "Lighting", 50, 0)]
    vectors = np.eye(2, dtype=np.float32)
    index = VectorIndex(products, vectors, np.zeros_like(vectors), [False, False],
                        {"schema_version": 1, "catalog_sha256": fingerprint(products),
                         "encoder": {"backend": "test-only"}})
    class TestEncoder:
        identity = {"backend": "test-only"}
        def text(self, texts):
            return np.array([[1, 0] if "jar" in t or "برطمان" in t else [0, 1] for t in texts])
        def images(self, paths):
            return np.array([[0, 1] for _ in paths])
    return SearchEngine(products, index, TestEncoder())


class RetrievalTests(unittest.TestCase):
    def test_semantic_branch_uses_query_vector(self):
        engine = fixture()
        self.assertEqual(engine.search("برطمان", mode="dense")["results"][0]["product"]["product_id"], "jar")
        self.assertEqual(engine.search(image=Path("unused-fixture"), mode="dense")["results"][0]["product"]["product_id"], "lamp")

    def test_filters_run_before_top_k(self):
        result = fixture().search("lamp", mode="dense", top_k=1, filters=Filters(in_stock=True))
        self.assertEqual(result["eligible_products"], 1)
        self.assertEqual(result["results"][0]["product"]["product_id"], "jar")
        self.assertEqual(fixture().search("jar", filters=Filters(max_price=1))["results"], [])

    def test_combined_query_and_weight_limits(self):
        self.assertEqual(fixture().search("jar", image=Path("fixture"), image_weight=1,
                                         mode="dense")["results"][0]["product"]["product_id"], "lamp")
        with self.assertRaises(ValueError):
            fixture().search("jar", image_weight=2)

    def test_zero_and_nonfinite_vectors_fail(self):
        with self.assertRaises(ValueError):
            normalize(np.array([[float("nan"), 1]]))
        engine = fixture()
        engine.encoder.text = lambda _: np.zeros((1, 2))
        with self.assertRaises(ValueError):
            engine.search("jar")

    def test_model_and_catalog_mismatch_are_rejected(self):
        engine = fixture()
        engine.encoder.identity = {"backend": "different"}
        with self.assertRaises(ValueError):
            engine.search("jar")
        with self.assertRaises(ValueError):
            SearchEngine(list(reversed(engine.products)), engine.index)

    def test_index_roundtrip_and_corruption(self):
        index = fixture().index
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"v1"
            index.save(target)
            loaded = VectorIndex.load(target)
            np.testing.assert_allclose(loaded.text_vectors, index.text_vectors)
            with self.assertRaises(ValueError):
                index.save(target)
            with (target/"vectors.npz").open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                VectorIndex.load(target)

    def test_invalid_search_inputs(self):
        for kwargs in [{"text": ""}, {"text": "jar", "top_k": 0},
                       {"text": "jar", "mode": "bad"},
                       {"image": Path("x"), "mode": "bm25"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fixture().search(**kwargs)
        with self.assertRaises(ValueError):
            Filters(min_price=50, max_price=20)
        with self.assertRaises(ValueError):
            Filters(max_price=float("nan"))


if __name__ == "__main__":
    unittest.main()
