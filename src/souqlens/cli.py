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


if __name__ == "__main__":
    main()
