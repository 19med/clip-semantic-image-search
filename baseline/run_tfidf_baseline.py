"""Run the caption TF-IDF baseline on a fixed COCO test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.bm25_baseline import CaptionTfidfRetriever
from src.data import load_coco_captions, load_image_ids, make_caption_queries, write_jsonl
from src.evaluate import evaluate_retrieval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True,
                        help="COCO captions_val2014.json path")
    parser.add_argument("--test-image-ids", required=True,
                        help="JSON list of Karpathy test image IDs")
    parser.add_argument("--output-dir", default="results/tfidf")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    test_ids = load_image_ids(args.test_image_ids)
    captions = load_coco_captions(args.annotations, test_ids)
    queries = make_caption_queries(captions)
    retriever = CaptionTfidfRetriever().fit(captions)
    results = retriever.retrieve_queries(queries, top_k=args.top_k)
    metrics = evaluate_retrieval(results)

    output_dir = Path(args.output_dir)
    write_jsonl(results, output_dir / "ranked_results.jsonl")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
