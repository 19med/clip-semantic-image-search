"""Encode COCO gallery images once and save an exact FAISS cosine index."""

import argparse
import json
from pathlib import Path

from run_clip import image_paths

MODEL_NAMES = {"ViT-B/32": "vit_b32", "ViT-B/16": "vit_b16"}


def build(args):
    import clip
    import faiss
    import numpy as np
    import torch
    from PIL import Image

    image_ids = json.loads(Path(args.ids).read_text(encoding="utf-8"))
    if not isinstance(image_ids, list) or len(image_ids) != len(set(image_ids)):
        raise ValueError("--ids must contain a list of unique image IDs")
    if len(image_ids) != 5000:
        print(f"Note: building for {len(image_ids)} images, not the full 5,000-image test gallery")
    paths = image_paths(args.image_root, image_ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            processed = []
            for path in paths[start:start + args.batch_size]:
                with Image.open(path) as im:
                    processed.append(preprocess(im.convert("RGB")))
            batch = torch.stack(processed).to(device)
            embeddings = model.encode_image(batch).float().cpu().numpy()
            batches.append(embeddings)
            print(f"Encoded {min(start + args.batch_size, len(paths))}/{len(paths)} images", flush=True)
    vectors = np.ascontiguousarray(np.concatenate(batches), dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    output_dir = Path(args.index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = MODEL_NAMES[args.model]
    faiss.write_index(index, str(output_dir / f"{slug}.faiss"))
    (output_dir / f"{slug}.json").write_text(
        json.dumps({"model": args.model, "image_ids": image_ids}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(image_ids)} image embeddings for {args.model} in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    build(args)
