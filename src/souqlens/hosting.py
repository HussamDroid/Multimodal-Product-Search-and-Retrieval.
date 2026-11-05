"""Package an index with its images and start the same search service on a host."""

import hashlib
import json
import logging
import os
from pathlib import Path, PureWindowsPath
import shutil
import tempfile

from .catalog import resolve_image
from .index import VectorIndex
from .ranker import PairwiseRanker


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_bundle(index_path: Path, image_root: Path, output: Path, ranker_path=None):
    if output.exists():
        raise ValueError("Bundle output already exists; use a new versioned directory")
    index = VectorIndex.load(index_path)
    if ranker_path:
        PairwiseRanker.load(ranker_path, index)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".souqlens-bundle-", dir=output.parent))
    try:
        (staging/"index").mkdir()
        (staging/"images").mkdir()
        for name in ["manifest.json", "vectors.npz"]:
            shutil.copy2(index_path/name, staging/"index"/name)
        for product in index.products:
            if not product.image:
                continue
            if Path(product.image).is_absolute() or PureWindowsPath(product.image).drive:
                raise ValueError("Hosted image paths must be relative; prepare the catalog before indexing")
            source = resolve_image(image_root, product.image)
            expected = index.metadata.get("image_sha256", {}).get(product.product_id)
            if not expected or sha256(source) != expected:
                raise ValueError(f"Image changed since indexing: {product.product_id}. Rebuild the index first.")
            relative = source.relative_to(image_root.resolve())
            target = staging/"images"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if ranker_path:
            shutil.copy2(ranker_path, staging/"ranker.json")
        files = {p.relative_to(staging).as_posix(): sha256(p)
                 for p in staging.rglob("*") if p.is_file()}
        (staging/"bundle.json").write_text(json.dumps({"schema_version": 1,
            "catalog_sha256": index.metadata["catalog_sha256"], "products": len(index.products),
            "has_ranker": bool(ranker_path), "files": files}, indent=2), encoding="utf-8")
        validate_bundle(staging)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"products": len(index.products), "output": str(output),
            "catalog_sha256": index.metadata["catalog_sha256"], "files": len(files)}


def validate_bundle(directory: Path):
    root = directory.resolve()
    manifest = json.loads((root/"bundle.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported deployment bundle version")
    declared = manifest.get("files", {})
    if not {"index/manifest.json", "index/vectors.npz"}.issubset(declared):
        raise ValueError("Bundle is missing its declared index files")
    for relative, expected in declared.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Bundle file is missing or its checksum differs: {relative}")
    index = VectorIndex.load(root/"index")
    if index.metadata["catalog_sha256"] != manifest["catalog_sha256"] or len(index.products) != manifest["products"]:
        raise ValueError("Bundle catalog identity does not match its index")
    for product in index.products:
        if product.image:
            path = resolve_image(root/"images", product.image)
            relative = path.relative_to(root).as_posix()
            if relative not in declared:
                raise ValueError("Product image is not declared in the deployment bundle")
    if manifest.get("has_ranker"):
        if "ranker.json" not in declared:
            raise ValueError("Bundle is missing its declared reranker")
        PairwiseRanker.load(root/"ranker.json", index)
    return manifest


def stage_space(repo_root: Path, output: Path):
    if output.exists():
        raise ValueError("Space source output already exists")
    required = {"pyproject.toml": repo_root/"pyproject.toml",
                "Dockerfile": repo_root/"deploy/Dockerfile",
                "README.md": repo_root/"deploy/README.md"}
    if not all(path.is_file() for path in required.values()) or not (repo_root/"src/souqlens").is_dir():
        raise ValueError("Run from the SouqLens repository or provide --repo-root")
    if output.resolve().is_relative_to((repo_root/"src").resolve()):
        raise ValueError("Stage deployment files outside the source directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".souqlens-space-", dir=output.parent))
    try:
        for target, source in required.items():
            shutil.copy2(source, staging/target)
        for source in (repo_root/"src/souqlens").rglob("*"):
            if source.is_file() and source.suffix in {".py", ".html"} and "__pycache__" not in source.parts:
                target = staging/"src"/source.relative_to(repo_root/"src")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output": str(output), "note": "Source only; no catalogs, indexes or images were copied"}


def main():
    logging.basicConfig(level=logging.INFO)
    bundle_dir = os.environ.get("SOUQLENS_BUNDLE_DIR")
    dataset_id = os.environ.get("SOUQLENS_DATASET_ID")
    if bundle_dir:
        directory = Path(bundle_dir)
    elif dataset_id:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset",
            revision=os.environ.get("SOUQLENS_DATASET_REVISION"),
            local_dir=tempfile.mkdtemp(prefix="souqlens-dataset-")))
    else:
        raise RuntimeError("Configure SOUQLENS_BUNDLE_DIR or SOUQLENS_DATASET_ID; no sample fallback is enabled")
    manifest = validate_bundle(directory)
    from .cli import indexed_engine
    from .api import create_app
    import uvicorn
    engine = indexed_engine(directory/"index", os.environ.get("SOUQLENS_DEVICE", "cpu"))
    logging.info("Loading query models for %s products", manifest["products"])
    # Load before accepting requests so the first visitor does not trigger model initialization.
    engine._encoder()
    ranker = PairwiseRanker.load(directory/"ranker.json", engine.index) if manifest.get("has_ranker") else None
    app = create_app(engine, directory/"images", ranker)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
