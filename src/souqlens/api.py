"""Local search API. Deploy behind authentication before exposing private catalogs."""

from io import BytesIO
import logging
from pathlib import Path
import tempfile
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .catalog import resolve_image
from .search import Filters

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class TextSearch(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    mode: Literal["bm25", "dense", "hybrid"] = "bm25"
    top_k: int = Field(default=10, ge=1, le=100)
    min_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    in_stock: bool = False
    category: str | None = None
    rerank: bool = False


def create_app(engine, image_root: Path, ranker=None):
    app = FastAPI(title="SouqLens", version="0.1.0",
                  description="English, Arabic and image product retrieval. Local demo; no authentication.")
    image_root = image_root.resolve()
    products = {p.product_id: p for p in engine.products}

    def filters(minimum, maximum, stock, category):
        try:
            return Filters(minimum, maximum, stock, category)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    def search(**kwargs):
        rerank = kwargs.pop("rerank", False)
        if rerank and ranker is None:
            raise HTTPException(422, "No trained reranker is configured")
        try:
            result = engine.search(**kwargs, ranker=ranker if rerank else None)
            for row in result["results"]:
                product = row["product"]
                product["image_url"] = f"/products/{product['product_id']}/image" if product["image"] else None
            return result
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            logging.getLogger(__name__).exception("Embedding backend unavailable")
            raise HTTPException(503, "Embedding backend unavailable. Check model installation and server logs.") from error

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (Path(__file__).parent/"web/index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "products": len(products),
                "modes": ["bm25", "dense", "hybrid"] if engine.index else ["bm25"],
                "model_loaded": engine.encoder is not None,
                "encoder": engine.index.metadata["encoder"] if engine.index else None,
                "reranker": ranker is not None,
                "categories": sorted({p.category for p in products.values() if p.category}),
                "note": "An index being present does not verify the query encoder can load."}

    @app.get("/products/{product_id}/image")
    def image(product_id: str):
        product = products.get(product_id)
        if not product or not product.image:
            raise HTTPException(404, "Product image not found")
        try:
            path = resolve_image(image_root, product.image)
        except ValueError as error:
            raise HTTPException(404, "Product image not available") from error
        if not path.is_file():
            raise HTTPException(404, "Product image not available")
        return FileResponse(path)

    @app.post("/search/text")
    def text_search(request: TextSearch):
        return search(text=request.text, mode=request.mode, top_k=request.top_k,
                      filters=filters(request.min_price, request.max_price, request.in_stock, request.category),
                      rerank=request.rerank)

    @app.post("/search/image")
    async def image_search(
        file: Annotated[UploadFile, File()],
        text: Annotated[str, Form(max_length=2000)] = "",
        mode: Annotated[Literal["dense", "hybrid"], Form()] = "hybrid",
        top_k: Annotated[int, Form(ge=1, le=100)] = 10,
        min_price: Annotated[float | None, Form(ge=0)] = None,
        max_price: Annotated[float | None, Form(ge=0)] = None,
        in_stock: Annotated[bool, Form()] = False,
        category: Annotated[str | None, Form()] = None,
        rerank: Annotated[bool, Form()] = False,
        image_weight: Annotated[float, Form(ge=0, le=1)] = 0.65,
    ):
        payload = await file.read(MAX_UPLOAD_BYTES+1)
        await file.close()
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the 10 MB upload limit")
        constraints = filters(min_price, max_price, in_stock, category)
        try:
            with Image.open(BytesIO(payload)) as picture:
                if picture.width * picture.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Image exceeds the 20 megapixel limit")
                picture.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
            raise HTTPException(422, "Upload a valid image file") from error
        with tempfile.TemporaryDirectory(prefix="souqlens-query-") as directory:
            path = Path(directory)/"query.image"
            path.write_bytes(payload)
            return await run_in_threadpool(search, text=text, image=path, mode=mode, top_k=top_k,
                                          filters=constraints, rerank=rerank, image_weight=image_weight)

    return app
