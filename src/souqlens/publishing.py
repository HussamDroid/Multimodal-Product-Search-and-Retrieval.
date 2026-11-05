"""Publish staged source to an existing Gradio Space without changing its hardware."""

import os
from pathlib import Path
import re


def validate_space_source(source: Path):
    if not source.is_dir():
        raise ValueError("Staged source directory does not exist")
    required = {"app.py", "README.md", "requirements.txt", "pyproject.toml"}
    for name in required:
        if not (source/name).is_file():
            raise ValueError(f"Staged source is missing {name}; run stage-space --sdk zerogpu")
    if "sdk: gradio" not in (source/"README.md").read_text(encoding="utf-8"):
        raise ValueError("The staged README must declare the Gradio SDK")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in staged deployment source")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        allowed = relative.as_posix() in required or (
            relative.parts[0] == "src" and path.suffix in {".py", ".html"})
        if not allowed:
            raise ValueError(f"Unexpected file in deployment source: {relative}. Catalog data must be uploaded separately.")


def publish_space(source: Path, space_id: str, source_commit="local", api=None):
    validate_space_source(source)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", space_id):
        raise ValueError("Provide a Space ID in account/name format")
    if api is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Add HF_TOKEN as a GitHub Actions repository secret before deploying")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    info = api.space_info(space_id)
    if info.sdk != "gradio":
        raise ValueError("The target Space must use Gradio; its SDK was not changed")
    runtime = api.get_space_runtime(space_id)
    hardware = {str(getattr(value, "value", value)) for value in
                [runtime.hardware, runtime.requested_hardware] if value is not None}
    if not hardware or hardware != {"zero-a10g"}:
        raise ValueError("The target Space must already have ZeroGPU selected. Hardware and billing settings were not changed.")
    commit = api.upload_folder(repo_id=space_id, repo_type="space", folder_path=str(source),
                              parent_commit=info.sha,
                              commit_message=f"Deploy SouqLens source {source_commit[:12]}")
    return {"space_id": space_id, "source_commit": source_commit,
            "space_commit": commit.oid, "status": "source_uploaded",
            "note": "Source upload succeeded. Check the Space build and runtime before calling it deployed."}
