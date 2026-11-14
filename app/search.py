
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_DIR,
    MODEL_DIR,
    COLLECTION_NAME,
    TOP_K,
)


class KnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer(
            str(MODEL_DIR),
            device="cpu",
            local_files_only=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )

    def search(self, question: str, k: int = TOP_K):
        if not question.strip():
            raise ValueError("Вопрос не должен быть пустым")

        count = self.collection.count()
        if count == 0:
            return []

        embedding = self.model.encode(
            ["query: " + question],
            normalize_embeddings=True,
        )

        results = self.collection.query(
            query_embeddings=embedding.tolist(),
            n_results=min(k, count),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        hits = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": float(distance),
            })

        return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k должно быть >= 1")

    kb = KnowledgeBase()
    hits = kb.search(args.question, args.top_k)

    for index, hit in enumerate(hits, 1):
        print(f"\n{'=' * 60}")
        print(f"Результат №{index}")
        print(f"Источник: {hit['source']}")
        print(f"Страница: {hit['page']}")
        print(f"Cosine distance: {hit['distance']:.4f}")
        print("-" * 60)
        print(hit["text"])
