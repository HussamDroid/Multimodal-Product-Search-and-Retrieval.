import importlib.util
from pathlib import Path
import tempfile
import unittest

from souqlens.gradio_demo import DemoService, create_demo
from souqlens.hosting import resolve_bundle_directory, stage_space
from test_retrieval import fixture


class GradioDemoTests(unittest.TestCase):
    def test_adapter_runs_search_and_returns_availability_without_stock_counts(self):
        service = DemoService(fixture(), Path("."))
        gallery, rows, summary = service.search("jar", None, "dense", None, 30, True, "All categories", False, 5)
        self.assertEqual(rows[0][0], "jar")
        self.assertEqual(rows[0][5], "In stock")
        self.assertEqual(gallery, [])
        self.assertIn("Queue time is excluded", summary)

    def test_invalid_modes_and_missing_ranker_are_explicit(self):
        service = DemoService(fixture(), Path("."))
        with self.assertRaisesRegex(ValueError, "reference image"):
            service.search("jar", "photo.jpg", "bm25", None, None, False, None, False, 5)
        with self.assertRaisesRegex(ValueError, "No trained reranker"):
            service.search("jar", None, "hybrid", None, None, False, None, True, 5)

    def test_bundle_configuration_is_required(self):
        with self.assertRaisesRegex(RuntimeError, "no sample fallback"):
            resolve_bundle_directory({})

    def test_zerogpu_staging_has_gradio_files_without_docker_or_private_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["pyproject.toml", "deploy/zerogpu/app.py", "deploy/zerogpu/README.md",
                         "deploy/zerogpu/requirements.txt", "src/souqlens/api.py", "data/private/master.csv"]:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fixture")
            stage_space(root, root/"staged", "zerogpu")
            self.assertTrue((root/"staged/app.py").is_file())
            self.assertTrue((root/"staged/requirements.txt").is_file())
            self.assertFalse((root/"staged/Dockerfile").exists())
            self.assertFalse((root/"staged/data").exists())

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio extra not installed")
    def test_gradio_setup_and_configured_layouts_construct(self):
        self.assertIsNotNone(create_demo())
        self.assertIsNotNone(create_demo(DemoService(fixture(), Path("."))))

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio extra not installed")
    def test_rendered_default_prices_include_the_full_catalog(self):
        service = DemoService(fixture(), Path("."))
        demo = create_demo(service)
        components = demo.get_config_file()["components"]
        prices = {c["props"]["label"]: c["props"]["value"] for c in components
                  if c.get("props", {}).get("label") in {"Minimum price QAR", "Maximum price QAR"}}
        minimum, maximum = prices["Minimum price QAR"], prices["Maximum price QAR"]
        constraints = service.validate("jar", None, "bm25", minimum, maximum, False,
                                       "All categories", False, 5)
        self.assertTrue(all(constraints.allows(p) for p in service.engine.products))
        _, rows, summary = service.search("jar", None, "bm25", minimum, maximum, False,
                                          "All categories", False, 5)
        self.assertTrue(rows)
        self.assertIn(f"{len(service.engine.products)} eligible products", summary)
        # An explicitly entered zero maximum must still filter out paid products.
        constraints = service.validate("jar", None, "bm25", 0, 0, False,
                                       "All categories", False, 5)
        self.assertFalse(any(constraints.allows(p) for p in service.engine.products if p.price > 0))
