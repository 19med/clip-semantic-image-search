# CLIP Semantic Image Search

This repository contains the caption-based retrieval baseline and shared evaluation contract for the final project.

## Your workstream: caption TF-IDF baseline

The baseline treats each image's human captions as indexed documents. A text query is compared with every caption using TF-IDF cosine similarity; the image score is the maximum score among its captions. For a test caption query, its exact indexed caption is left out to avoid self-matching leakage. This is a lexical baseline, unlike CLIP, which compares text features directly with visual image features.

## Data contract

Use **MS COCO 2014 captions** and the approved Karpathy test image IDs. Place the captions annotation JSON outside Git, for example:

```text
data/raw/annotations/captions_val2014.json
data/splits/karpathy_test_image_ids.json
```

The split file must be a JSON array of COCO image IDs. Do not generate a random split.

## Download and prepare only the files needed for the baseline

The lexical baseline needs captions and split metadata, not image pixels. From the project root:

```bash
mkdir -p data/raw/annotations data/raw/karpathy
curl -L http://images.cocodataset.org/annotations/annotations_trainval2014.zip \
  -o data/raw/annotations/annotations_trainval2014.zip
unzip data/raw/annotations/annotations_trainval2014.zip \
  'annotations/captions_val2014.json' -d data/raw

curl -L 'https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download=' \
  -o data/raw/karpathy/dataset_coco.json
python prepare_karpathy_split.py \
  --karpathy-json data/raw/karpathy/dataset_coco.json
```

This creates `data/splits/karpathy_test_image_ids.json`, containing the fixed 5,000-image test set. The project intentionally does not download COCO image files for this baseline; the captions are sufficient. Your teammate will need the images for CLIP embedding extraction.

## Installation and baseline run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python run_tfidf_baseline.py \
  --annotations data/raw/annotations/captions_val2014.json \
  --test-image-ids data/splits/karpathy_test_image_ids.json \
  --output-dir results/tfidf
```

The run writes `ranked_results.jsonl` and `metrics.json`. The JSONL record format is shared with the CLIP pipeline:

```json
{"query_id": 0, "query_text": "a dog running", "ground_truth_image_id": 123, "ranked_image_ids": [123, 456], "scores": [0.91, 0.63]}
```

## Current next steps

1. Add the official Karpathy test image-ID file.
2. Run the baseline and save metrics.
3. Select representative failure cases from its ranked-results file.
4. Build the web UI against this same results contract; later replace its retrieval source with the best CLIP API.
