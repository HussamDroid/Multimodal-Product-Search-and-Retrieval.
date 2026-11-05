"""Publish staged source to an existing Gradio Space without changing its hardware."""

import os
from pathlib import Path
import re
import time


def hub_api():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Add HF_TOKEN as a GitHub Actions repository secret before deploying")
    from huggingface_hub import HfApi
    return HfApi(token=token)


def wait_for_space(space_id, expected_commit, timeout=600, interval=10, api=None, clock=None, sleeper=None):
    if timeout <= 0 or interval <= 0:
        raise ValueError("Timeout and poll interval must be positive")
    api = api if api is not None else hub_api()
    clock, sleeper = clock or time.monotonic, sleeper or time.sleep
    deadline = clock()+timeout
    last_stage = "UNKNOWN"
    while True:
        info = api.space_info(space_id)
        if info.sha != expected_commit:
            raise RuntimeError("The Space source changed during verification; verify the latest source commit instead")
        runtime = api.get_space_runtime(space_id)
        last_stage = str(getattr(runtime.stage, "value", runtime.stage))
        if last_stage == "RUNNING":
            return {"status": "running", "stage": last_stage, "space_commit": expected_commit}
        if last_stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "PAUSED", "DELETING"}:
            raise RuntimeError(f"Space stopped at {last_stage}. Inspect the Space build/runtime logs.")
        if clock() >= deadline:
            raise RuntimeError(f"Space did not become RUNNING within {timeout} seconds; last stage: {last_stage}")
        sleeper(min(interval, max(0, deadline-clock())))


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


def publish_space(source: Path, space_id: str, source_commit="local", api=None, wait=False, timeout=600):
    validate_space_source(source)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", space_id):
        raise ValueError("Provide a Space ID in account/name format")
    if api is None:
        api = hub_api()
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
    result = {"space_id": space_id, "source_commit": source_commit,
            "space_commit": commit.oid, "status": "source_uploaded",
            "note": "Source upload succeeded. Check the Space build and runtime before calling it deployed."}
    if wait:
        print(f"Source uploaded to {space_id}; waiting for its runtime.", flush=True)
        result.update(wait_for_space(space_id, commit.oid, timeout=timeout, api=api))
        result["note"] = "The Space is RUNNING. Query inference still needs testing with a configured catalog."
    return result
