from io import BytesIO
from pathlib import Path
import unittest

from PIL import Image
from fastapi.testclient import TestClient

from souqlens.api import create_app
from test_retrieval import fixture


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(fixture(), Path("data/sample")))

    def test_health_and_demo(self):
        self.assertEqual(self.client.get("/health").json()["modes"], ["bm25", "dense", "hybrid"])
        self.assertIn("SouqLens", self.client.get("/").text)

    def test_search_and_filters(self):
        response = self.client.post("/search/text", json={"text": "jar", "mode": "dense", "max_price": 30})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["product"]["product_id"], "jar")
        self.assertEqual(self.client.post("/search/text", json={"text": "jar", "top_k": 100}).status_code, 200)

    def test_invalid_inputs_and_unavailable_ranker(self):
        for body in [{"text": "jar", "top_k": 0}, {"text": "jar", "mode": "unknown"},
                     {"text": "jar", "min_price": 50, "max_price": 10},
                     {"text": "jar", "rerank": True}, {"text": "   "}]:
            self.assertEqual(self.client.post("/search/text", json=body).status_code, 422)

    def test_image_and_combined_search(self):
        image = BytesIO()
        Image.new("RGB", (16, 16), "blue").save(image, format="PNG")
        response = self.client.post("/search/image", data={"text": "lamp", "mode": "dense"},
                                    files={"file": ("query.png", image.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["product"]["product_id"], "lamp")

    def test_invalid_and_oversized_uploads(self):
        response = self.client.post("/search/image", files={"file": ("fake.jpg", b"not an image", "image/jpeg")})
        self.assertEqual(response.status_code, 422)
        response = self.client.post("/search/image", files={"file": ("big.jpg", b"0"*(10*1024*1024+1), "image/jpeg")})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.get("/products/missing/image").status_code, 404)


if __name__ == "__main__":
    unittest.main()
