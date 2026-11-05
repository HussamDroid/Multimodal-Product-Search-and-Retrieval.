# Host a real catalog

The hosted service needs a prepared catalog bundle plus the application source. A Windows drive path such as `Z:\Mini Mall\Mini-Mall-Images` is local to the office and cannot be used directly by a cloud server.

The full-model container is `deploy/Dockerfile`. The root Dockerfile remains the lightweight BM25 sample demo. The hosted container loads models before accepting requests and refuses to start without an explicit bundle. It does not silently substitute synthetic products.

## 1. Prepare and verify the catalog locally

Follow [REAL_CATALOG.md](REAL_CATALOG.md), then build `artifacts/real-index-v1`. Review the import report and test several English, Arabic and photo queries. Do not reuse the synthetic reranker with a real index; it is tied to its training catalog.

Export the index with its referenced images:

```powershell
cd D:\souqlens
.\.venv\Scripts\souqlens.exe export-bundle --index artifacts/real-index-v1 --image-root "Z:\Mini Mall\Mini-Mall-Images" --output artifacts/hosted-bundle-v1
```

If a compatible reranker was trained on this catalog, add `--ranker artifacts/real-ranker-v1.json`. Export validates image checksums against the index and copies only referenced covers. If a source image changed, rebuild the index first.

The bundle contains product fields and stock quantities, image files, vectors, model identity and checksums. Keep it out of the public Git repository. This packaging step does not upload anything.

## 2. Choose a host

### Hugging Face Docker Space

Suitable for an ML portfolio demo. Current official documentation says creating a compute Space (including Docker) requires a paid account plan; CPU Basic is listed as 2 vCPU / 16 GB RAM with no hourly compute charge. Hardware, account eligibility and costs must be confirmed in the account before creation. The code never purchases a plan or requests upgraded hardware.

Start with a private Space and private dataset. A public app would expose catalog fields and images through the API; private storage alone does not make a public app private.

1. Create a private dataset repository in the chosen Hugging Face account for the catalog bundle.
2. Upload the contents of `artifacts/hosted-bundle-v1` to the root of that dataset. Keep `bundle.json`, `index/` and `images/` together.
3. Record the resulting dataset commit hash.
4. Prepare source-only deployment files:

```powershell
.\.venv\Scripts\souqlens.exe stage-space --output artifacts/space-source-v1
```

5. Create a private Space with Docker as the SDK and the chosen CPU hardware. Upload the contents of `artifacts/space-source-v1` to its root. This staged directory contains application source, a Dockerfile and the Space README metadata; it contains no product data.
6. In the Space settings, configure:

| Setting | Type | Value |
| --- | --- | --- |
| `SOUQLENS_DATASET_ID` | Variable | Account/dataset repository ID |
| `SOUQLENS_DATASET_REVISION` | Variable | Exact uploaded dataset commit hash |
| `HF_TOKEN` | Secret | Read-scoped token able to access the private dataset |

Enter tokens directly into the provider's secret settings, never source code or chat. The app downloads the pinned dataset snapshot at startup and validates its checksums. Writable local disk is treated as a cache, not the authoritative copy of the catalog. Model and bundle downloads can make a cold start slower.

7. Wait for the build and model initialization. Check `/health`, then test the UI with real English, Arabic and image queries. A successful local test does not prove hosted deployment succeeded.

Once the correct account, audience and hardware are chosen, deployment can be configured to mirror future source checkpoints. No automatic source-to-Space sync is configured yet.

### Existing VPS or another Docker host

Use the same container with the prepared bundle mounted read-only. On the host, with the bundle at `/srv/souqlens/bundle`:

```bash
docker build -f deploy/Dockerfile -t souqlens:hosted .
docker run --rm -p 127.0.0.1:7860:7860 \
  -e SOUQLENS_BUNDLE_DIR=/catalog \
  --mount type=bind,src=/srv/souqlens/bundle,dst=/catalog,readonly \
  souqlens:hosted
```

Point the host's authenticated HTTPS reverse proxy at port 7860. The app itself has no authentication or rate limiting. Do not expose a private catalog directly without the intended access control. A normal shared-web-hosting plan is not the same as a Docker-capable VPS; confirm the actual hosting plan and available RAM before selecting it.

## Runtime settings

| Variable | Behavior |
| --- | --- |
| `SOUQLENS_BUNDLE_DIR` | Load a mounted local bundle; takes precedence over dataset download |
| `SOUQLENS_DATASET_ID` | Download a bundle from a Hugging Face dataset if no local bundle is set |
| `SOUQLENS_DATASET_REVISION` | Pin the dataset commit used by the service |
| `SOUQLENS_DEVICE` | Default `cpu`; use `cuda` only with compatible hardware/image |
| `PORT` | Default 7860 |

## Verification status

Importer, bundle validation and source-only staging have automated tests. The hosted Dockerfile has not been built in this workspace because Docker is unavailable. No real catalog has been supplied for this deployment, and no hosting account has been selected or changed. There is no live real-product URL yet.

Official references:

- https://huggingface.co/docs/hub/spaces-sdks-docker
- https://huggingface.co/docs/hub/spaces-overview
