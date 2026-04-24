# Drift from the official JEPA line (I-JEPA / V-JEPA)

This document lists intentional differences between `jepacluster` and the official JEPA work (for example [I-JEPA, arXiv:2301.08243](https://arxiv.org/pdf/2301.08243) and the [V-JEPA codebase](https://github.com/facebookresearch/jepa)). The goal is to separate **modality** choices from **scope** choices.

## What drifts because this is logs, not images or video

- **Input unit:** Logs are tokenized text lines in a sliding window, not image patches or spatio-temporal tubes. There is no 2D/3D patch grid, RGB pixels, or frame stack.
- **Backbone:** A line-level transformer over sequence positions replaces a Vision Transformer (or video ViT) over patch tokens. Line embeddings pool token embeddings per line instead of using a standard ViT patch embedder.
- **Masking:** Masks are defined over *log lines* (contiguous spans in a window), not multi-block spatial masks (I-JEPA) or tube-style video masks (V-JEPA). What counts as "context" vs "target" is sequential, not spatially local in an image.
- **Targets:** The model predicts latent vectors for *hidden lines*, not for held-out image/video regions. Semantics are textual and procedural (events, templates) rather than visual appearance or motion.
- **No pixel or frame decoder:** Like JEPA, there is no reconstruction of raw pixels; unlike video demos, there is also no separate generative decoder for visualization—only latent prediction for representation learning on text.

## What drifts due to a desire for simplicity, this being a POC, lack of compute, etc.

- **Scale:** Smaller models, shorter schedules, and single-machine (often single-GPU) training instead of large distributed runs and very long iteration budgets.
- **Optimizer and training recipe:** Adam with a compact cosine/warmup schedule, rather than the full LARS/Lion-scale recipes and hyperparameter grids used in reference pretraining.
- **Masking richness:** A simpler mask sampler (mostly contiguous spans, a small number of masks per step) instead of the aggressive, multi-block random masking strategies tuned for vision SSL.
- **Loss and stabilization:** Smooth L1 in latent space with an EMA teacher, without the full set of normalization or auxiliary objectives sometimes used at large scale; collapse is monitored lightly rather than with a full suite of regularizers.
- **Data and evaluation pipeline:** Prototype-oriented data loading (e.g. directory of log files, fixed windowing) and clustering downstream, rather than ImageNet/Kinetics-scale datasets and standardized probe suites from the papers.
- **Configuration surface:** Fewer knobs and ablations than the official config trees; defaults favor getting an end-to-end run on modest hardware.
- **Predictor vs encoder:** The predictor is enforced to be **strictly smaller** than the encoder (fewer transformer layers, narrower feed-forward than the encoder blocks, fewer attention heads where possible), aligning with the reference *idea* while staying small overall.

If a drift belongs in both sections, it is listed under **logs vs images/video** when the change is forced by the modality; otherwise it is listed under **simplicity / POC / compute**.
