# Visual pipeline

> Part of the DineWise project plan. Visual-summary retrieval, CLIP zero-shot, fusion, and limits.
> See [the plan index](README.md) for the full reading order.

---

## 13. Visual pipeline

Because the dataset contains no image files, the visual experiment is split into two honest components.

### 13.1 Component A — text-to-visual retrieval

- Index the `photo_summay.csv` text with BGE.
- Queries such as "romantic interior", "outdoor patio", "bright and modern" retrieve visual descriptions.
- Results map to restaurants via `business_id`.
- Metadata: `document_type=visual`, plus `venue_style`, `indoor_color_tone`, `target_audience`.
- Evaluation: 20 visual queries with manually identified gold restaurants, using the photo summary text as the relevance source. Metrics: Success@5, Precision@5, MRR.
- Claim: this improves retrieval for ambience queries where structured attributes are sparse. It is **not** image understanding.

### 13.2 Component B — image-to-cuisine via CLIP zero-shot

Model: `openai/clip-vit-base-patch32`.

Process:

1. Build 101 text prompts from Food-101 class names: `"a photo of {class}, a type of food"`.
2. Compute CLIP text embeddings for all prompts.
3. Encode the uploaded image with the CLIP image encoder.
4. Compute cosine similarity; take the top-5 food classes and their probabilities.
5. Map food classes to cuisine families with a hand-built table (e.g., pizza → Italian, sushi → Japanese, tacos → Mexican, paella → Spanish, dumplings → Chinese).
6. Use the mapped cuisine as a **soft** filter or query expansion term.
7. Show the top-3 predicted classes in the UI with confidence, so the user can see the uncertainty.

Rules:

- Never claim that the image is from a specific restaurant.
- Never use the image to assert food quality or authenticity.
- If the top-1 probability is below a threshold (calibrated on a validation subset), do not apply a cuisine filter.
- Images are processed in memory and not stored unless the user opts in.

Evaluation:

- Download a stratified subset of Food-101 validation images via streaming (10 per class = 1,010 images) rather than the full 5 GB dataset.
- Measure top-1 and top-5 **class** accuracy.
- Measure mapped-cuisine accuracy on classes with a confident mapping.
- Report coverage: how many of the 101 classes have an unambiguous cuisine mapping.

### 13.3 Fusion

For a visual query:

- Component A contributes restaurant candidates directly.
- Component B contributes a cuisine hint only.
- Fuse with the text retrieval results using the same RRF procedure, with the visual list treated as another ranked list.
- The final answer must state that visual matching is based on descriptions or predicted food class, not on actual restaurant photos.

### 13.4 Metrics and limitations

Metrics: visual Success@5, visual Precision@5, visual MRR, CLIP top-1/top-5 class accuracy, mapped-cuisine accuracy, visual retrieval latency, and the effect on end-to-end nDCG.

Limitations to report:

- no ground-truth restaurant images,
- photo summaries are GPT-4o generated,
- Food-101 classes do not cover all cuisines,
- cuisine inference from a single dish is weak evidence,
- visual similarity does not establish restaurant identity, authenticity, or quality.

---
