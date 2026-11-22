"""COCO-caption data loading and project interchange formats."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_coco_captions(annotation_path: str | Path, image_ids: set[int] | None = None) -> dict[int, list[str]]:
    """Return ``{image_id: [caption, ...]}`` from a COCO captions annotation JSON file."""
    with Path(annotation_path).open(encoding="utf-8") as handle:
        annotations = json.load(handle)["annotations"]

    captions: dict[int, list[str]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_ids is None or image_id in image_ids:
            captions[image_id].append(annotation["caption"].strip())
    return dict(captions)


def load_image_ids(path: str | Path) -> set[int]:
    """Load a JSON list of COCO image IDs."""
    with Path(path).open(encoding="utf-8") as handle:
        return {int(image_id) for image_id in json.load(handle)}


def make_caption_queries(captions_by_image: dict[int, list[str]]) -> list[dict]:
    """Turn each ground-truth caption into one retrieval query."""
    queries = []
    query_id = 0
    for image_id in sorted(captions_by_image):
        for caption_index, caption in enumerate(captions_by_image[image_id]):
            queries.append(
                {
                    "query_id": query_id,
                    "query_text": caption,
                    "ground_truth_image_id": image_id,
                    # Prevent the lexical baseline from matching a query to itself.
                    "excluded_caption_index": caption_index,
                }
            )
            query_id += 1
    return queries


def write_jsonl(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
