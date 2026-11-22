"""Caption-based TF-IDF image retrieval baseline.

Every image is represented by its five captions. At query time, each caption is
scored separately and an image receives the maximum score of its captions.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class CaptionTfidfRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2))
        self.caption_matrix = None
        self.caption_image_ids: np.ndarray | None = None
        self.caption_indices: np.ndarray | None = None
        self.image_ids: np.ndarray | None = None

    def fit(self, captions_by_image: dict[int, list[str]]) -> "CaptionTfidfRetriever":
        caption_texts: list[str] = []
        caption_image_ids: list[int] = []
        caption_indices: list[int] = []
        for image_id in sorted(captions_by_image):
            for caption_index, caption in enumerate(captions_by_image[image_id]):
                caption_texts.append(caption)
                caption_image_ids.append(image_id)
                caption_indices.append(caption_index)
        if not caption_texts:
            raise ValueError("No captions were provided.")
        self.caption_matrix = self.vectorizer.fit_transform(caption_texts)
        self.caption_image_ids = np.asarray(caption_image_ids, dtype=np.int64)
        self.caption_indices = np.asarray(caption_indices, dtype=np.int64)
        self.image_ids = np.unique(self.caption_image_ids)
        return self

    def search(
        self, query: str, top_k: int = 10, exclude_caption: tuple[int, int] | None = None
    ) -> list[tuple[int, float]]:
        if self.caption_matrix is None or self.caption_image_ids is None or self.caption_indices is None or self.image_ids is None:
            raise RuntimeError("Call fit before search.")
        query_vector = self.vectorizer.transform([query])
        caption_scores = (self.caption_matrix @
                          query_vector.T).toarray().ravel()

        best_by_image: dict[int, float] = defaultdict(float)
        for image_id, caption_index, score in zip(
            self.caption_image_ids, self.caption_indices, caption_scores, strict=True
        ):
            if exclude_caption == (int(image_id), int(caption_index)):
                continue
            best_by_image[int(image_id)] = max(
                best_by_image[int(image_id)], float(score))
        ranked = sorted(best_by_image.items(),
                        key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]

    def retrieve_queries(self, queries: list[dict], top_k: int = 10) -> list[dict]:
        results = []
        for query in queries:
            excluded_caption = None
            if "excluded_caption_index" in query:
                excluded_caption = (int(query["ground_truth_image_id"]), int(
                    query["excluded_caption_index"]))
            ranked = self.search(
                query["query_text"], top_k=top_k, exclude_caption=excluded_caption)
            results.append(
                {
                    **query,
                    "ranked_image_ids": [image_id for image_id, _ in ranked],
                    "scores": [score for _, score in ranked],
                }
            )
        return results
