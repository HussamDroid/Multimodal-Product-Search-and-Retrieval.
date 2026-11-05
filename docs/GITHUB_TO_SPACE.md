# Deploy GitHub source to Hussam07/souqlens

Target Space: https://huggingface.co/spaces/Hussam07/souqlens

The public API requires authentication for this Space, so its SDK and hardware could not be verified without account access. The deploy command checks both before uploading: it requires the existing Gradio SDK and ZeroGPU hardware (`zero-a10g`). It never creates a Space, changes visibility, changes hardware, or purchases compute.

## One-time setup

1. In the Hugging Face account, verify that `Hussam07/souqlens` uses **Gradio** and the **free ZeroGPU** option. If ZeroGPU is not available, stop and resolve eligibility rather than selecting paid hardware.
2. Open https://huggingface.co/settings/tokens and create a fine-grained token with write access limited to this Space. Use a deployment-specific name such as `souqlens-github-deploy`. Do not paste it into chat or commit it to the repository.
3. In the GitHub project, open **Settings → Secrets and variables → Actions → New repository secret**. Name it **`HF_TOKEN`** and paste the deployment token directly into the secret value.
4. Open **Actions → Deploy SouqLens to Hugging Face → Run workflow**, using the `main` branch.

The default Space ID is already `Hussam07/souqlens`. An optional repository variable `HF_SPACE_ID` can change it later, but no additional variable is needed for this target.

The workflow stages only the Gradio source, checks the existing SDK/hardware and uploads a source commit. It uses the current Space commit as a parent guard so concurrent edits do not get silently overwritten. It does not delete unrelated files already present in the Space.

## After upload

The workflow reports a Space commit hash once the source is uploaded, then waits up to ten minutes for that exact commit to reach `RUNNING`. Build/runtime failures, paused states, a changed source commit or a timeout fail the workflow. A running setup screen is not proof of model inference: test queries after configuring the real catalog.

Until the real-catalog dataset is configured, the app intentionally displays **Catalog not configured** and disables search. Follow [ZEROGPU.md](ZEROGPU.md) to connect the checksummed product bundle using Space variables and a separate read-scoped dataset token. The GitHub deployment token does not need access to product datasets.

## Future checkpoints

After `HF_TOKEN` is added, relevant pushes to `main` trigger the source deployment workflow. Before the secret is configured, push-triggered runs skip uploading and show a notice. A manually requested run without the secret fails with an explanation.

The first manually triggered source upload completed successfully through GitHub Actions. Runtime verification was then added to check private Space startup using the existing secret. Publisher tests cover SDK/hardware validation, the source allowlist, parent-commit guard, startup states and bounded timeouts. No credential value is read by the agent or included in source. Real-catalog ZeroGPU inference still needs verification after a product bundle is supplied.

Official references:

- https://huggingface.co/docs/hub/spaces-github-actions
- https://huggingface.co/docs/hub/security-tokens
