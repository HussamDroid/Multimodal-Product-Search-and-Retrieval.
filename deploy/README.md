---
title: SouqLens
emoji: 🔎
colorFrom: pink
colorTo: gray
sdk: docker
app_port: 7860
---

# SouqLens

English, Arabic and reference-image product retrieval. Source development and checkpoints are maintained in the [GitHub repository](https://github.com/HussamDroid/Multimodal-Product-Search-and-Retrieval.).

This deployment requires a prepared catalog bundle. Configure `SOUQLENS_DATASET_ID` and a read-scoped `HF_TOKEN` secret for the private bundle dataset, or mount a bundle and set `SOUQLENS_BUNDLE_DIR` on a Docker host. Pin `SOUQLENS_DATASET_REVISION` to the uploaded dataset commit.

Private is the default deployment audience. The app serves the indexed product fields and images, including stock quantities. Review the catalog and intended audience before changing a deployment to public.
