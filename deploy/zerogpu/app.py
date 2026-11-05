"""Gradio Space entry point. Import spaces before the ML stack initializes CUDA."""

import spaces
import os
from pathlib import Path
import sys

# The Space installs dependencies before copying the application source.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from souqlens.gradio_demo import create_demo, load_service

# Model placement happens at module level as required by the ZeroGPU runtime.
if os.environ.get("SOUQLENS_BUNDLE_DIR") or os.environ.get("SOUQLENS_DATASET_ID"):
    SERVICE, BUNDLE = load_service(device="cuda")
else:
    SERVICE, BUNDLE = None, None


@spaces.GPU(duration=15)
def semantic_search(*args):
    if SERVICE is None:
        raise ValueError("No catalog is configured")
    return SERVICE.search(*args)


demo = create_demo(SERVICE, semantic_search=semantic_search)

if __name__ == "__main__":
    allowed = [str((BUNDLE/"images").resolve())] if BUNDLE else []
    blocked = [str((BUNDLE/name).resolve()) for name in ["index", "bundle.json", "ranker.json"]] if BUNDLE else []
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860,
                                  allowed_paths=allowed, blocked_paths=blocked,
                                  max_file_size="10mb", show_error=False)
