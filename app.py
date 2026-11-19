"""Local CLIP image-search API and webpage. Run `python app.py --help`."""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from build_index import MODEL_NAMES
from run_clip import image_paths


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    model: Literal["ViT-B/32", "ViT-B/16"] = "ViT-B/16"
    top_k: int = Field(default=10, ge=1, le=30)


class SearchService:
    def __init__(self, image_root, ids_path, index_dir):
        self.index_dir = Path(index_dir)
        self.ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        if not isinstance(self.ids, list) or len(self.ids) != len(set(self.ids)):
            raise ValueError("Image ID file must be a list of unique IDs")
        image_paths_in_order = image_paths(image_root, self.ids)
        self.paths = dict(zip(self.ids, image_paths_in_order))

    @lru_cache(maxsize=2)
    def load_model(self, name):
        import clip
        import faiss
        import torch

        slug = MODEL_NAMES[name]
        meta_path = self.index_dir / f"{slug}.json"
        index_path = self.index_dir / f"{slug}.faiss"
        if not meta_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"Build the {name} index before searching")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("model") != name or metadata.get("image_ids") != self.ids:
            raise ValueError(f"Index for {name} was built with different image IDs")
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(self.ids):
            raise ValueError(f"Index for {name} contains {index.ntotal} images; expected {len(self.ids)}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load(name, device=device)
        model.eval()
        return index, model, device

    def available_models(self):
        return [name for name, slug in MODEL_NAMES.items()
                if (self.index_dir / f"{slug}.faiss").is_file()
                and (self.index_dir / f"{slug}.json").is_file()]

    def search(self, query, name, top_k):
        import clip
        import faiss
        import numpy as np
        import torch

        index, model, device = self.load_model(name)
        with torch.no_grad():
            tokens = clip.tokenize([query], truncate=True).to(device)
            vector = model.encode_text(tokens).float().cpu().numpy()
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        faiss.normalize_L2(vector)
        scores, positions = index.search(vector, min(top_k, len(self.ids)))
        return [{"image_id": self.ids[int(position)], "score": round(float(score), 4),
                 "image_url": f"/api/images/{self.ids[int(position)]}"}
                for score, position in zip(scores[0], positions[0])]


def create_app(service):
    api = FastAPI(title="CLIP Semantic Image Search")
    webpage = Path(__file__).parent / "web" / "index.html"

    @api.get("/")
    def home():
        return FileResponse(webpage, media_type="text/html")

    @api.get("/api/health")
    def health():
        return {"gallery_images": len(service.ids), "available_models": service.available_models()}

    @api.post("/api/search")
    def search(request: SearchRequest):
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="Enter a description to search")
        try:
            results = service.search(query, request.model, request.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"query": query, "model": request.model, "gallery_images": len(service.ids),
                "results": results}

    @api.get("/api/images/{image_id}")
    def image(image_id: int):
        path = service.paths.get(image_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Image not in the selected gallery")
        return FileResponse(path, media_type="image/jpeg")

    return api


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(SearchService(args.image_root, args.ids, args.index_dir)),
                host="127.0.0.1", port=args.port)
