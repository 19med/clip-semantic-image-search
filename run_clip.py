"""Run CLIP text-to-image retrieval against the same query set as a caption baseline.

The input images are COCO JPEGs named like COCO_val2014_000000000042.jpg.
The JSONL input uses query_text, query_id, and ground_truth_image_id.
"""

import argparse
import json
from pathlib import Path


def read_inputs(image_ids_file, queries_file, limit_images=None, limit_queries=None):
    image_ids = json.loads(Path(image_ids_file).read_text(encoding="utf-8"))
    if not isinstance(image_ids, list) or len(image_ids) != len(set(image_ids)):
        raise ValueError("Image IDs must be a list without duplicates")
    if limit_images:
        image_ids = image_ids[:limit_images]
    gallery = set(image_ids)
    queries = []
    with Path(queries_file).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["ground_truth_image_id"] in gallery:
                queries.append({
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "ground_truth_image_id": row["ground_truth_image_id"],
                })
                if limit_queries and len(queries) >= limit_queries:
                    break
    if not queries:
        raise ValueError("No queries have a correct image in the selected gallery")
    return image_ids, queries


def image_paths(root, image_ids):
    """Find COCO 2014 JPGs by their numeric IDs; fail on missing/duplicate IDs."""
    wanted = set(image_ids)
    paths = {}
    for path in Path(root).rglob("*.jpg"):
        if not path.stem.startswith("COCO_"):
            continue
        tail = path.stem.rsplit("_", 1)[-1]
        if not tail.isdigit():
            continue
        image_id = int(tail)
        if image_id in wanted:
            if image_id in paths:
                raise ValueError(f"Two files found for image ID {image_id}")
            paths[image_id] = path
    missing = [i for i in image_ids if i not in paths]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(image_ids)} images. "
            f"First missing IDs: {missing[:5]}. Check --image-root."
        )
    return [paths[i] for i in image_ids]


def recall_counts(rows):
    n = len(rows)
    return {
        "recall_at_1": sum(r["ground_truth_image_id"] in r["ranked_image_ids"][:1] for r in rows) / n,
        "recall_at_5": sum(r["ground_truth_image_id"] in r["ranked_image_ids"][:5] for r in rows) / n,
        "recall_at_10": sum(r["ground_truth_image_id"] in r["ranked_image_ids"][:10] for r in rows) / n,
        "num_queries": n,
    }


def run(args):
    image_ids, queries = read_inputs(args.ids, args.queries, args.limit_images, args.limit_queries)
    paths = image_paths(args.image_root, image_ids)
    # Import model dependencies only after verifying that all input images exist.
    import clip
    import faiss
    import numpy as np
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)
    model.eval()
    image_vectors = []
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            batch = []
            for path in paths[start:start + args.batch_size]:
                with Image.open(path) as im:
                    batch.append(preprocess(im.convert("RGB")))
            vectors = model.encode_image(torch.stack(batch).to(device)).float()
            image_vectors.append(vectors.cpu().numpy())
            print(f"Encoded {min(start + args.batch_size, len(paths))}/{len(paths)} images", flush=True)
    image_vectors = np.ascontiguousarray(np.concatenate(image_vectors), dtype=np.float32)
    faiss.normalize_L2(image_vectors)
    index = faiss.IndexFlatIP(image_vectors.shape[1])
    index.add(image_vectors)

    rows = []
    with torch.no_grad():
        for start in range(0, len(queries), args.batch_size):
            batch = queries[start:start + args.batch_size]
            tokens = clip.tokenize([r["query_text"] for r in batch], truncate=True).to(device)
            vectors = model.encode_text(tokens).float().cpu().numpy()
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors)
            scores, positions = index.search(vectors, min(10, len(image_ids)))
            for row, score_row, pos_row in zip(batch, scores, positions):
                rows.append({
                    **row,
                    "model": args.model,
                    "ranked_image_ids": [image_ids[int(p)] for p in pos_row],
                    "scores": [float(s) for s in score_row],
                })
            if (start // args.batch_size) % 20 == 0:
                print(f"Searched {min(start + args.batch_size, len(queries))}/{len(queries)} queries", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = {"model": args.model, "gallery_images": len(image_ids), **recall_counts(rows)}
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved rankings to {output} and metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, help="Directory containing COCO JPEG images")
    parser.add_argument("--ids", required=True, help="JSON list of gallery image IDs")
    parser.add_argument("--queries", required=True, help="JSONL rows with query_text and ground_truth_image_id")
    parser.add_argument("--model", choices=["ViT-B/32", "ViT-B/16"], default="ViT-B/32")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-images", type=int, help="Use the first N images for a quick check")
    parser.add_argument("--limit-queries", type=int, help="Use the first N matching queries")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_images is not None and args.limit_images < 1) or (args.limit_queries is not None and args.limit_queries < 1):
        parser.error("Batch size and limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
