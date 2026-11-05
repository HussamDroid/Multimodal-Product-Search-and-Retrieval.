"""Command-line entry points for experiments and local development."""

import argparse
import json
from pathlib import Path

from .catalog import load_catalog
from .lexical import BM25


def main(argv=None):
    parser = argparse.ArgumentParser(prog="souqlens")
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("baseline", help="Run the BM25 baseline without model downloads")
    search.add_argument("--catalog", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    build = sub.add_parser("index", help="Encode a catalog with the pretrained multilingual CLIP pair")
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--image-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--device", default="cpu")
    build.add_argument("--text-revision")
    build.add_argument("--image-revision")
    dense = sub.add_parser("search", help="Search an existing vector index")
    dense.add_argument("--index", type=Path, required=True)
    dense.add_argument("--query", default="")
    dense.add_argument("--image", type=Path)
    dense.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    dense.add_argument("--top-k", type=int, default=5)
    dense.add_argument("--min-price", type=float)
    dense.add_argument("--max-price", type=float)
    dense.add_argument("--in-stock", action="store_true")
    dense.add_argument("--category")
    dense.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "baseline":
        if not args.query.strip() or not 1 <= args.top_k <= 100:
            parser.error("Provide a query and a top-k between 1 and 100")
        products = load_catalog(args.catalog)
        scores = BM25([p.text for p in products]).scores(args.query)
        order = sorted(range(len(products)), key=lambda i: (-float(scores[i]), products[i].product_id))
        print(json.dumps([{"product_id": products[i].product_id, "name": products[i].name_en,
                           "score": float(scores[i])} for i in order[:args.top_k] if scores[i] > 0],
                         ensure_ascii=False, indent=2))
    elif args.command == "index":
        from .encoders import ClipEncoder
        from .index import VectorIndex
        products = load_catalog(args.catalog, args.image_root)
        encoder = ClipEncoder(device=args.device, text_revision=args.text_revision,
                              image_revision=args.image_revision)
        index = VectorIndex.build(products, args.image_root, encoder)
        index.save(args.output)
        print(json.dumps(index.metadata, indent=2))
    elif args.command == "search":
        from .encoders import ClipEncoder
        from .index import VectorIndex
        from .search import Filters, SearchEngine
        index = VectorIndex.load(args.index)
        meta = index.metadata["encoder"]
        engine = SearchEngine(index.products, index=index, encoder_factory=lambda: ClipEncoder(
            device=args.device, text_revision=meta["text_revision"], image_revision=meta["image_revision"]))
        result = engine.search(args.query, image=args.image, mode=args.mode, top_k=args.top_k,
                               filters=Filters(args.min_price, args.max_price, args.in_stock, args.category))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
