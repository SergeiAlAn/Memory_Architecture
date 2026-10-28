"""Пункт 2 ДЗ: загрузка векторных данных о Солнце и планетах в Pinecone."""

from __future__ import annotations

import time

from data import SOLAR_SYSTEM_PHRASES
from embeddings import EmbeddingClient
from pine import PineconeClient

# Границы фраз по объектам в SOLAR_SYSTEM_PHRASES (start inclusive, end exclusive).
OBJECT_RANGES: list[tuple[str, int, int]] = [
    ("sun", 0, 12),
    ("mercury", 12, 23),
    ("venus", 23, 34),
    ("earth", 34, 45),
    ("mars", 45, 56),
    ("jupiter", 56, 67),
    ("saturn", 67, 78),
    ("uranus", 78, 89),
    ("neptune", 89, 100),
]

OBJECT_BY_INDEX: dict[int, str] = {}
for name, start, end in OBJECT_RANGES:
    for i in range(start, end):
        OBJECT_BY_INDEX[i] = name

EMBED_BATCH = 50
UPSERT_BATCH = 50


def main() -> None:
    phrases = SOLAR_SYSTEM_PHRASES
    if len(phrases) < 20:
        raise RuntimeError("Нужно не менее 20 фраз для пункта 2")

    pine = PineconeClient(index_name="embeddings", dimension=1536)
    embedder = EmbeddingClient(model="text-embedding-3-small", dimension=1536)

    print(f"Фраз к загрузке: {len(phrases)}")
    print(f"Индекс: {pine.index_name}")

    all_vectors: list[dict] = []
    for start in range(0, len(phrases), EMBED_BATCH):
        chunk = phrases[start : start + EMBED_BATCH]
        print(f"Эмбеддинг фраз {start + 1}–{start + len(chunk)}...")
        vectors = embedder.embed_many(chunk)
        for offset, (text, values) in enumerate(zip(chunk, vectors)):
            idx = start + offset
            vector_id = f"solar-{idx:03d}"
            all_vectors.append(
                {
                    "id": vector_id,
                    "values": values,
                    "metadata": {
                        "text": text,
                        "object": OBJECT_BY_INDEX[idx],
                        "topic": "solar_system",
                        "index": idx,
                    },
                }
            )

    upserted_total = 0
    for start in range(0, len(all_vectors), UPSERT_BATCH):
        batch = all_vectors[start : start + UPSERT_BATCH]
        result = pine.upsert(batch)
        upserted_total += result["upserted_count"]
        print(f"Upsert {start + 1}–{start + len(batch)}: {result}")

    # Небольшая пауза: serverless Pinecone иногда обновляет stats с задержкой.
    time.sleep(2)

    sample_ids = ["solar-000", "solar-034", "solar-099"]
    fetched = pine.fetch(sample_ids)
    print("\nПроверка выборки:")
    for vector_id in sample_ids:
        record = fetched.get(vector_id)
        if not record:
            print(f"  [!] не найден: {vector_id}")
            continue
        meta = record["metadata"]
        print(f"  {vector_id} | {meta.get('object')} | {meta.get('text', '')[:70]}...")

    stats = pine.stats()
    print(f"\nUpsert всего: {upserted_total}")
    print(f"Статистика индекса: total_vector_count={stats.get('total_vector_count')}")
    print("Пункт 2 выполнен успешно.")


if __name__ == "__main__":
    main()
