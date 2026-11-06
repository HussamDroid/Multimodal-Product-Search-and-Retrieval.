"""Gradio presentation layer shared by local and ZeroGPU demos."""

from pathlib import Path

from .catalog import resolve_image
from .hosting import resolve_bundle_directory, validate_bundle
from .ranker import PairwiseRanker
from .search import Filters

RESULT_HEADERS = ["Product ID", "English name", "Arabic name", "Category", "Price QAR",
                  "Availability", "Rank score", "BM25", "Cosine"]


class DemoService:
    def __init__(self, engine, image_root, ranker=None):
        self.engine, self.image_root, self.ranker = engine, Path(image_root), ranker

    def search(self, text, image, mode, min_price, max_price, in_stock, category, rerank, top_k):
        constraints = self.validate(text, image, mode, min_price, max_price, in_stock, category, rerank, top_k)
        result = self.engine.search(text=text or "", image=Path(image) if image else None,
            mode=mode, top_k=int(top_k), filters=constraints,
            ranker=self.ranker if rerank else None)
        gallery, rows = [], []
        for row in result["results"]:
            product = row["product"]
            if product["image"]:
                path = resolve_image(self.image_root, product["image"])
                gallery.append((str(path), f"{product['name_en']} · QAR {product['price']:.2f}"))
            rows.append([product["product_id"], product["name_en"], product["name_ar"],
                product["category"], product["price"], "In stock" if product["stock"] > 0 else "Out of stock",
                round(row["score"], 5), round(row["bm25"], 4),
                round(row["cosine"], 4) if row["cosine"] is not None else None])
        summary = (f"{len(rows)} results · {result['eligible_products']} eligible products · "
                   f"{result['latency_ms']:.1f} ms search time. "
                   "Queue time is excluded. Scores are ranking values, not confidence probabilities.")
        return gallery, rows, summary

    def validate(self, text, image, mode, min_price, max_price, in_stock, category, rerank, top_k):
        if not str(text or "").strip() and not image:
            raise ValueError("Enter a description or upload a reference image")
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval method")
        if not 1 <= int(top_k) <= 20:
            raise ValueError("Choose between 1 and 20 results")
        if image and mode == "bm25":
            raise ValueError("Select Dense or Hybrid when providing a reference image")
        if rerank and self.ranker is None:
            raise ValueError("No trained reranker is configured for this catalog")
        return Filters(min_price, max_price, in_stock,
                       category if category and category != "All categories" else None)


def load_service(device="cpu", environ=None):
    from .cli import indexed_engine
    directory = resolve_bundle_directory(environ)
    validate_bundle(directory)
    engine = indexed_engine(directory/"index", device)
    # ZeroGPU requires CUDA placement during startup, before decorated request functions run.
    engine._encoder()
    ranker = PairwiseRanker.load(directory/"ranker.json", engine.index) if (directory/"ranker.json").is_file() else None
    return DemoService(engine, directory/"images", ranker), directory


def create_demo(service=None, semantic_search=None):
    import gradio as gr
    maximum_catalog_price = max((p.price for p in service.engine.products), default=0) if service else 0
    categories = sorted({p.category for p in service.engine.products if p.category}) if service else []

    def dispatch(text, image, mode, minimum, maximum, stock, category, rerank, count):
        if service is None:
            raise gr.Error("No catalog bundle is configured")
        try:
            service.validate(text, image, mode, minimum, maximum, stock, category, rerank, count)
            callback = service.search if mode == "bm25" else semantic_search or service.search
            return callback(text, image, mode, minimum, maximum, stock, category, rerank, count)
        except ValueError as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title="SouqLens · Multimodal product search") as demo:
        gr.Markdown("# SouqLens\nSearch products in English, Arabic or with a reference photo.")
        if service is None:
            gr.Markdown("**Catalog not configured.** Add the catalog dataset settings or a local bundle. "
                        "No real products have been loaded, and no synthetic catalog is substituted.")
        else:
            gr.Markdown(f"**{len(service.engine.products)} products indexed.** "
                        "Semantic requests use shared GPU resources on ZeroGPU and can queue.")
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(label="Description · English or Arabic", lines=3,
                                  placeholder="Small glass jars for kitchen storage")
                image = gr.Image(label="Reference photo (optional)", type="filepath", sources=["upload"],
                                 interactive=service is not None)
                mode = gr.Dropdown(choices=[("Hybrid", "hybrid"), ("Dense", "dense"), ("BM25 baseline", "bm25")],
                                   value="hybrid", label="Retrieval method")
                with gr.Row():
                    minimum = gr.Number(value=0, label="Minimum price QAR", minimum=0)
                    # Explicit bounds keep the initial search inclusive when the browser sends zero for an empty number.
                    maximum = gr.Number(value=maximum_catalog_price, label="Maximum price QAR", minimum=0)
                category = gr.Dropdown(choices=["All categories"]+categories,
                                       value="All categories", label="Category")
                stock = gr.Checkbox(label="In-stock products only")
                rerank = gr.Checkbox(label="Apply trained reranker", interactive=bool(service and service.ranker))
                count = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Results")
                submit = gr.Button("Search products", variant="primary", interactive=service is not None)
            with gr.Column(scale=2):
                status = gr.Textbox(label="Search status", interactive=False)
                gallery = gr.Gallery(label="Matched products", columns=3, object_fit="contain")
                table = gr.Dataframe(headers=RESULT_HEADERS, label="Ranking details", interactive=False)
        submit.click(dispatch, inputs=[text, image, mode, minimum, maximum, stock, category, rerank, count],
                     outputs=[gallery, table, status], concurrency_limit=1)
        gr.Markdown("Product embeddings are prepared offline. This demo encodes queries at request time. "
                    "Price and stock are explicit metadata filters; image/text similarity does not guarantee a match.")
    return demo
