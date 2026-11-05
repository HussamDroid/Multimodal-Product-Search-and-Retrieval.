---
title: SouqLens
emoji: 🔎
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: 3.12.12
app_file: app.py
---

# SouqLens — multilingual product retrieval

English, Arabic, image and combined-query retrieval with a lexical baseline and optional trained reranker.

Select **ZeroGPU** hardware for the free deployment route on an eligible personal account. A Gradio SDK declaration alone does not select hardware or establish free-tier eligibility. Do not select paid hardware to follow this deployment guide.

Set `SOUQLENS_DATASET_ID`, pin `SOUQLENS_DATASET_REVISION`, and add a read-scoped `HF_TOKEN` secret for a private bundle dataset. Until a bundle is configured, the app displays a setup message with search disabled.

Repository and checkpoints: https://github.com/HussamDroid/Multimodal-Product-Search-and-Retrieval.
