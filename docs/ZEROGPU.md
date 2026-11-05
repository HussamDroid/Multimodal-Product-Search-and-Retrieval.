# Free Gradio / ZeroGPU deployment

The source includes a Gradio interface and a separate ZeroGPU entry point. The existing FastAPI and Docker options remain available.

## Eligibility first

According to Hugging Face's current ZeroGPU documentation, a free personal account in good standing with a verified email and an account age above 30 days can host up to two ZeroGPU Spaces. This is an exception to the paid-plan requirement for ordinary compute Space creation.

An account existing does not by itself establish eligibility. Before deploying, verify that the account offers **ZeroGPU** hardware with the free option. If it does not, stop and check eligibility rather than selecting paid CPU/GPU hardware or subscribing. Usage quotas and shared queues apply.

Reference: https://huggingface.co/docs/hub/spaces-zerogpu

## Prepare the real catalog

Complete [REAL_CATALOG.md](REAL_CATALOG.md), build the real index, and export a deployment bundle using [HOSTING.md](HOSTING.md). Product embeddings are prepared offline. Only query inference runs in the hosted demo.

The private dataset layout is unchanged:

```text
bundle.json
index/manifest.json
index/vectors.npz
images/<barcode>/1.jpg
ranker.json                 optional, trained on this exact index
```

Keep real product exports and bundles outside the public source repository. The Gradio result table displays availability rather than exact stock quantities; the private dataset still contains the indexed product records.

## Stage the Gradio source

In PowerShell:

```powershell
cd D:\souqlens
git pull origin main
.\.venv\Scripts\python.exe -m pip install -e ".[gradio]"
.\.venv\Scripts\souqlens.exe stage-space --sdk zerogpu --output artifacts/zerogpu-source-v1
```

The staged directory contains `app.py`, `README.md`, `requirements.txt`, `pyproject.toml` and application source. It includes no catalog data, model weights or Dockerfile. Use a new output directory for each stage operation.

## Create and configure the Space

1. Use the intended personal Hugging Face account. Create a **Gradio** Space named `souqlens` (or another available name) and select the **free ZeroGPU** hardware option. Verify its availability before committing to creation. A Gradio Space on ordinary CPU hardware is not the same option.
2. Keep the Space private for initial setup where the account supports that audience. If private ZeroGPU hosting is unavailable, do not change it to public without deciding which product data may be shown.
3. Upload the contents of `artifacts/zerogpu-source-v1` to the Space repository root. The staged README already specifies the Gradio SDK, supported Python version and application entry point. It does not automatically assign hardware.
4. Upload the prepared bundle to a private dataset owned by the chosen account, then record that dataset commit hash.
5. Configure the Space settings:

| Setting | Type | Value |
| --- | --- | --- |
| `SOUQLENS_DATASET_ID` | Variable | Account/dataset repository ID |
| `SOUQLENS_DATASET_REVISION` | Variable | Exact dataset commit hash |
| `HF_TOKEN` | Secret | Read-scoped access to the private dataset |

Enter the token in the provider's secret settings, not source code or chat. The account and product files have not been supplied to this workspace, so no Space or dataset has been created by this checkpoint.

6. Wait for the build and model loading. Test English, Arabic, image-only and combined queries. Compare keyword/dense/hybrid methods and verify price and availability filters. A URL or successful source upload alone is not proof that GPU inference works.

With no bundle settings, the app shows a setup message and disables search. An invalid configured bundle fails validation; synthetic products are never substituted silently.

## Runtime behavior

- `spaces` is imported before the model stack initializes CUDA.
- Query models are placed on CUDA during module-level startup, following the ZeroGPU runtime's emulation model.
- The semantic callback requests a GPU using `@spaces.GPU(duration=15)`. Keyword-only BM25 calls do not enter this callback.
- Input validation runs before requesting GPU resources.
- Query-image uploads are limited to 10 MB. Results expose referenced product images, while the index files, bundle manifest and reranker are blocked from Gradio file serving.
- Displayed search latency excludes shared-GPU queue time. Ranking values are not calibrated probabilities.

The adapter and UI construction are tested locally. Live ZeroGPU scheduling, account eligibility and a real-product deployment remain unverified until the target Space is available.

## Optional local CPU check

With an existing bundle and ML dependencies installed:

```powershell
.\.venv\Scripts\souqlens.exe serve-gradio --bundle artifacts/hosted-bundle-v1
```

Open http://127.0.0.1:7860. This uses CPU locally and does not need the `spaces` package or a Hugging Face subscription. It does not test ZeroGPU scheduling.
