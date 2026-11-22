"""Create COCO image-ID files from Karpathy's dataset_coco.json split file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--karpathy-json", required=True,
                        help="Path to dataset_coco.json")
    parser.add_argument("--output-dir", default="data/splits")
    args = parser.parse_args()

    with Path(args.karpathy_json).open(encoding="utf-8") as handle:
        images = json.load(handle)["images"]

    split_map = {"train": ["train", "restval"],
                 "val": ["val"], "test": ["test"]}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for output_name, accepted_splits in split_map.items():
        image_ids = sorted(
            int(image["cocoid"])
            for image in images
            if image["split"] in accepted_splits and "cocoid" in image
        )
        with (output_dir / f"karpathy_{output_name}_image_ids.json").open("w", encoding="utf-8") as handle:
            json.dump(image_ids, handle)
        print(f"{output_name}: {len(image_ids)} image IDs")


if __name__ == "__main__":
    main()
