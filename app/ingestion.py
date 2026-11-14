
import hashlib
import logging
import re
from pathlib import Path

import chromadb
import pymupdf
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer

from app.config import (
    DOCUMENTS_DIR,
    CHROMA_DIR,
    MODEL_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(directory: Path) -> list[dict]:
    documents = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".pdf":
            with pymupdf.open(path) as pdf:
                for page_number, page in enumerate(pdf, 1):
                    text = clean_text(
                        page.get_text("text", sort=True)
                    )
                    if text:
                        documents.append({
                            "text": text,
                            "source": str(
                                path.relative_to(directory)
                            ),
                            "page": page_number,
                        })

        elif path.suffix.lower() in (".txt", ".md"):
            text = clean_text(
                path.read_text(encoding="utf-8-sig")
            )
            if text:
                documents.append({
                    "text": text,
                    "source": str(
                        path.relative_to(directory)
                    ),
                    "page": 1,
                })

    return documents


def make_chunks(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for document in documents:
        parts = splitter.split_text(document["text"])

        for index, part in enumerate(parts):
            raw_id = (
                f"{document['source']}:"
                f"{document['page']}:{index}:{part}"
            )
            chunk_id = hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()

            chunks.append({
                "id": chunk_id,
                "text": part,
                "metadata": {
                    "source": document["source"],
                    "page": document["page"],
                    "chunk_index": index,
                },
            })

    return chunks


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Загрузка документов")
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        logger.warning("Документы не найдены")
        return

    chunks = make_chunks(documents)

    logger.info("Документов/страниц: %d", len(documents))
    logger.info("Чанков: %d", len(chunks))

    model = SentenceTransformer(
        str(MODEL_DIR),
        device="cpu",
        local_files_only=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={
            "hnsw": {"space": "cosine"}
        },
    )

    # Для учебного проекта перестраиваем индекс
    # целиком, чтобы не сохранять устаревшие чанки.
    client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={
            "hnsw": {"space": "cosine"}
        },
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            "passage: " + item["text"]
            for item in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[
                item["text"] for item in batch
            ],
            metadatas=[
                item["metadata"] for item in batch
            ],
            embeddings=embeddings.tolist(),
        )

        logger.info(
            "Проиндексировано: %d/%d",
            min(start + BATCH_SIZE, len(chunks)),
            len(chunks),
        )

    logger.info(
        "Готово. В базе %d фрагментов.",
        collection.count(),
    )


if __name__ == "__main__":
    main()