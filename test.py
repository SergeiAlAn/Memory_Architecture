"""Пункт 1 ДЗ: индекс Pinecone + embedding-модель + тест создания вектора."""

from embeddings import EmbeddingClient
from pine import PineconeClient

TEXT = "Пицца Пепперони"
VECTOR_ID = "hw1-test-pizza"


def main() -> None:
    # 1. Создать / подключить индекс
    pine = PineconeClient(index_name="embeddings", dimension=1536)
    print(f"[ok] Индекс: {pine.index_name}")
    print(f"     dimension={pine.dimension}, metric={pine.metric}")

    # 2. Подключить embedding-модель
    embedder = EmbeddingClient(model="text-embedding-3-small", dimension=1536)
    print(f"[ok] Embedding-модель: {embedder.model}")

    # 3. Создать вектор из текста и записать в Pinecone
    values = embedder.embed(TEXT)
    print(f"[ok] Вектор создан: len={len(values)}, sample={values[:3]}")

    upserted = pine.upsert(
        [
            {
                "id": VECTOR_ID,
                "values": values,
                "metadata": {"text": TEXT, "source": "homework-1"},
            }
        ]
    )
    print(f"[ok] Upsert: {upserted}")

    # Проверка: прочитать вектор обратно
    fetched = pine.fetch([VECTOR_ID])
    record = fetched.get(VECTOR_ID)
    if not record:
        raise RuntimeError(f"Вектор {VECTOR_ID} не найден после upsert")

    print(f"[ok] Fetch: id={record['id']}")
    print(f"     metadata={record['metadata']}")
    print(f"     values_len={len(record['values'])}")
    print(f"     stats={pine.stats()}")
    print("Пункт 1 выполнен успешно.")


if __name__ == "__main__":
    main()
