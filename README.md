# CLIP Semantic Image Search

A text-to-image retrieval project using **COCO 2014** images. We compare a caption-based **TF-IDF baseline** with two pretrained CLIP vision backbones, **ViT-B/32** and **ViT-B/16**. A local website lets users type a new description and view images retrieved by either CLIP model.

## How it works

1. CLIP encodes each image and the user's text into vectors (embeddings).
2. FAISS ranks images by cosine similarity to the text embedding.
3. The API returns the highest-ranked image IDs, scores and URLs to the website.

The TF-IDF baseline searches captions associated with gallery images using weighted word overlap. Its implementation, split-preparation script, and baseline documentation are included in the `baseline/` directory. The website currently supports live search with the two CLIP models.

## Test results

All reported runs searched the same **5,000-image COCO Karpathy test gallery** with **25,010 caption queries**. Recall@K is the percentage of queries for which the correct image appears in the first K results.

| Method | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| TF-IDF caption baseline¹ | 17.42% | 36.03% | 46.31% |
| CLIP ViT-B/32 | 30.45% | 55.97% | 66.86% |
| CLIP ViT-B/16 | **33.08%** | **58.36%** | **68.96%** |

¹ The TF-IDF scores were supplied by the other project member. For the final report, confirm that it used the same query IDs and image gallery and excluded the query caption itself from the searchable captions. CLIP read the supplied query texts and image IDs but calculated its **own** rankings. ViT-B/16 scored higher than ViT-B/32 at all three cutoffs.

The CLIP weights are pretrained; this project does not train them from scratch on COCO. The full training, validation and test split and each split's use still need to be documented in the final report. A 10-image smoke test was used to check the program; its scores are not included above.

## Model selection on validation

We selected the final CLIP backbone using the 5,000-image Karpathy validation split with 25,010 caption queries.

| Model | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| CLIP ViT-B/32 | 30.59% | 56.36% | 67.33% |
| CLIP ViT-B/16 | **32.91%** | **58.66%** | **69.61%** |

ViT-B/16 was selected because it achieved the best validation Recall@1, Recall@5, and Recall@10. Full results and systematic failure analysis are available in [docs/RESULTS.md](docs/RESULTS.md).

## Repository files

| File | Purpose |
| --- | --- |
| `run_clip.py` | Evaluate either model, save top-10 rankings and Recall@K metrics. |
| `build_index.py` | Encode the image gallery and save a reusable FAISS index. |
| `app.py` | Run the local API, webpage and image server. |
| `web/index.html` | Frontend that calls `/api/search`. |
| `LOCAL_APP_README.md` | Additional Windows instructions. |
| `.gitignore` | Keep image data, indexes and generated results out of Git. |

Images, captions, pretrained weights, indexes and results are **not bundled**. Obtain COCO 2014 images from the [COCO dataset](https://cocodataset.org/#download). The program expects names such as `COCO_val2014_000000000042.jpg` and scans the directory passed as `--image-root`.

## Run the website on Windows

### 1. Prepare files

Put `karpathy_test_image_ids.json` (a JSON list of 5,000 numeric IDs) and the COCO 2014 validation images alongside the project files. A typical layout is:

```text
CLIP_Part_2_Starter/
  app.py
  build_index.py
  run_clip.py
  karpathy_test_image_ids.json
  coco2014/images/val2014/COCO_val2014_000000000042.jpg
  web/index.html
```

### 2. Install dependencies

Use [PyTorch's install selector](https://pytorch.org/get-started/locally/) if you need a specific GPU build. In PowerShell:

```powershell
cd D:\CLIP_Part_2_Starter
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision pillow numpy faiss-cpu fastapi uvicorn
.\.venv\Scripts\python.exe -m pip install git+https://github.com/openai/CLIP.git
```

The CLIP installation command requires Git. First use of each model downloads its pretrained weights.

### 3. Build searchable indexes once

```powershell
.\.venv\Scripts\python.exe .\build_index.py --image-root .\coco2014 --ids .\karpathy_test_image_ids.json --model ViT-B/16 --index-dir .\indexes
.\.venv\Scripts\python.exe .\build_index.py --image-root .\coco2014 --ids .\karpathy_test_image_ids.json --model ViT-B/32 --index-dir .\indexes
```

The builder verifies that every ID has an image. Use the **same ID file** to launch the app; the API checks it against each saved index. The generated `indexes` directory is ignored by Git.

### 4. Start the app

```powershell
.\.venv\Scripts\python.exe .\app.py --image-root .\coco2014 --ids .\karpathy_test_image_ids.json --index-dir .\indexes
```

Open **http://127.0.0.1:8000** while PowerShell stays open; press **Ctrl+C** to stop. Do not open `web/index.html` as a `file:///` page because it needs the API. This is a local demo, not a public deployment.

## API

The frontend sends `POST /api/search` with JSON:

```json
{"query":"a dog sleeping on shoes","model":"ViT-B/16","top_k":10}
```

The response has `query`, `model`, `gallery_images`, and `results`. Each ranked result has `image_id`, cosine-similarity `score` and `image_url`. Scores are for ranking within a model; they are **not probabilities** or directly comparable to TF-IDF scores.

- `GET /api/health`: image count and indexed models.
- `GET /api/images/{image_id}`: an image from the selected gallery.
- `GET /docs`: interactive API documentation.

## Reproduce the CLIP evaluation

The query JSONL must contain `query_id`, `query_text` and `ground_truth_image_id` on each line. The shared TF-IDF `ranked_results.jsonl` contains these fields; CLIP reads its query information but does not use the TF-IDF ranks.

```powershell
.\.venv\Scripts\python.exe .\run_clip.py --image-root .\coco2014 --ids .\karpathy_test_image_ids.json --queries .\ranked_results.jsonl --model ViT-B/32 --output .\results\test_vit_b32.jsonl
.\.venv\Scripts\python.exe .\run_clip.py --image-root .\coco2014 --ids .\karpathy_test_image_ids.json --queries .\ranked_results.jsonl --model ViT-B/16 --output .\results\test_vit_b16.jsonl
```

Each command saves a rankings JSONL and a neighboring `.metrics.json`. The full test runs have already been completed; they do not need to be rerun for the website. The final report should add validation details, counted failure categories with image examples, and a discussion of limitations.

## References

- [COCO](https://cocodataset.org/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)
