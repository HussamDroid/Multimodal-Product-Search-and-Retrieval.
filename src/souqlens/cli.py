"""Command-line entry points for experiments and local development."""

import argparse
import json
from pathlib import Path

from .catalog import load_catalog
from .lexical import BM25


def indexed_engine(path, device="cpu"):
    from .encoders import ClipEncoder
    from .index import VectorIndex
    from .search import SearchEngine
    index = VectorIndex.load(path)
    meta = index.metadata["encoder"]
    return SearchEngine(index.products, index=index, encoder_factory=lambda: ClipEncoder(
        device=device, text_revision=meta["text_revision"], image_revision=meta["image_revision"]))


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
    dense.add_argument("--ranker", type=Path)
    evaluate = sub.add_parser("evaluate", help="Run held-out retrieval evaluation")
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--catalog", type=Path)
    source.add_argument("--index", type=Path)
    evaluate.add_argument("--queries", type=Path, required=True)
    evaluate.add_argument("--image-root", type=Path, required=True)
    evaluate.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    evaluate.add_argument("--split", choices=["train", "dev", "test"], default="test")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--ranker", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    train = sub.add_parser("train-ranker", help="Fit the linear pairwise reranker using train judgments only")
    train.add_argument("--index", type=Path, required=True)
    train.add_argument("--queries", type=Path, required=True)
    train.add_argument("--image-root", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=500)
    train.add_argument("--device", default="cpu")
    serve = sub.add_parser("serve", help="Start the local API and search demo")
    source = serve.add_mutually_exclusive_group(required=True)
    source.add_argument("--catalog", type=Path)
    source.add_argument("--index", type=Path)
    serve.add_argument("--image-root", type=Path, required=True)
    serve.add_argument("--ranker", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", default="cpu")
    prepare = sub.add_parser("prepare-catalog", help="Convert an Excel/CSV master export and match barcode cover images")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--image-root", type=Path, required=True)
    prepare.add_argument("--mapping", type=Path)
    prepare.add_argument("--sheet")
    bundle = sub.add_parser("export-bundle", help="Package a validated index and its images for hosting")
    bundle.add_argument("--index", type=Path, required=True)
    bundle.add_argument("--image-root", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument("--ranker", type=Path)
    stage = sub.add_parser("stage-space", help="Prepare source files for a Docker Space without copying private data")
    stage.add_argument("--repo-root", type=Path, default=Path("."))
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--sdk", choices=["docker", "zerogpu"], default="docker")
    gradio = sub.add_parser("serve-gradio", help="Test the Gradio interface locally on CPU")
    gradio.add_argument("--bundle", type=Path, required=True)
    gradio.add_argument("--host", default="127.0.0.1")
    gradio.add_argument("--port", type=int, default=7860)
    publish = sub.add_parser("publish-space", help="Upload source to an existing ZeroGPU Space using HF_TOKEN")
    publish.add_argument("--source", type=Path, required=True)
    publish.add_argument("--space-id", default="Hussam07/souqlens")
    publish.add_argument("--source-commit", default="local")
    publish.add_argument("--wait", action="store_true")
    publish.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.command == "publish-space":
        from .publishing import publish_space
        print(json.dumps(publish_space(args.source, args.space_id, args.source_commit,
                                       wait=args.wait, timeout=args.timeout), indent=2))
    elif args.command == "serve-gradio":
        from .gradio_demo import create_demo, load_service
        service, directory = load_service("cpu", {"SOUQLENS_BUNDLE_DIR": str(args.bundle)})
        create_demo(service).queue(max_size=20).launch(server_name=args.host, server_port=args.port,
            allowed_paths=[str((directory/"images").resolve())],
            blocked_paths=[str((directory/name).resolve()) for name in ["index", "bundle.json", "ranker.json"]],
            max_file_size="10mb", show_error=False)
    elif args.command == "stage-space":
        from .hosting import stage_space
        print(json.dumps(stage_space(args.repo_root, args.output, args.sdk), indent=2))
    elif args.command == "export-bundle":
        from .hosting import export_bundle
        print(json.dumps(export_bundle(args.index, args.image_root, args.output, args.ranker), indent=2))
    elif args.command == "prepare-catalog":
        from .importer import prepare_catalog
        mapping = json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else None
        print(json.dumps(prepare_catalog(args.source, args.output, args.image_root, mapping, args.sheet),
                         ensure_ascii=False, indent=2))
    elif args.command == "baseline":
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
        from .ranker import PairwiseRanker
        from .search import Filters, SearchEngine
        engine = indexed_engine(args.index, args.device)
        ranker = PairwiseRanker.load(args.ranker, engine.index) if args.ranker else None
        result = engine.search(args.query, image=args.image, mode=args.mode, top_k=args.top_k,
                               filters=Filters(args.min_price, args.max_price, args.in_stock, args.category), ranker=ranker)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command in {"evaluate", "train-ranker"}:
        from .evaluation import evaluate as run_evaluation, load_queries
        from .ranker import PairwiseRanker
        from .search import SearchEngine
        engine = indexed_engine(args.index, args.device) if args.index else SearchEngine(load_catalog(args.catalog))
        queries = load_queries(args.queries, engine.products)
        if args.command == "train-ranker":
            ranker = PairwiseRanker.train(engine, queries, args.image_root, steps=args.steps)
            ranker.save(args.output)
            print(json.dumps(ranker.metadata, indent=2))
        else:
            ranker = PairwiseRanker.load(args.ranker, engine.index) if args.ranker and engine.index else None
            if args.ranker and ranker is None:
                parser.error("A reranker requires --index")
            report = run_evaluation(engine, queries, image_root=args.image_root, mode=args.mode,
                                    split=args.split, k=args.top_k, ranker=ranker)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False, indent=2)
            print(json.dumps(report["summary"], indent=2))
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        from .ranker import PairwiseRanker
        from .search import SearchEngine
        engine = indexed_engine(args.index, args.device) if args.index else SearchEngine(load_catalog(args.catalog, args.image_root))
        if args.ranker and not engine.index:
            parser.error("A reranker requires --index")
        ranker = PairwiseRanker.load(args.ranker, engine.index) if args.ranker else None
        uvicorn.run(create_app(engine, args.image_root, ranker), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
