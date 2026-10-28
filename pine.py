"""Клиент для работы с векторной базой данных Pinecone."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


class PineconeClient:
    """Клиент для подключения, записи, чтения, поиска и удаления векторов в Pinecone.

    Attributes:
        api_key: API-ключ Pinecone.
        index_name: Имя индекса.
        dimension: Размерность векторов.
        metric: Метрика сходства (cosine, euclidean, dotproduct).
        cloud: Облачный провайдер для serverless-индекса.
        region: Регион serverless-индекса.
        pc: Экземпляр клиента Pinecone.
        index: Объект индекса для операций с векторами.
    """

    def __init__(
        self,
        index_name: str = "embeddings",
        dimension: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        api_key: str | None = None,
    ) -> None:
        """Инициализирует клиент и подключается к индексу Pinecone.

        Args:
            index_name: Имя индекса. По умолчанию ``"embeddings"``.
            dimension: Размерность векторов. Для ``text-embedding-3-small`` — 1536.
            metric: Метрика сходства. По умолчанию ``"cosine"``.
            cloud: Облако для serverless-индекса. По умолчанию ``"aws"``.
            region: Регион serverless-индекса. По умолчанию ``"us-east-1"``.
            api_key: API-ключ. Если не передан, читается из ``PINICONE_TOKEN``
                или ``PINECONE_API_KEY``.

        Returns:
            None.
        """
        self.api_key = api_key or os.getenv("PINICONE_TOKEN") or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API-ключ Pinecone не найден. "
                "Укажите api_key или задайте PINICONE_TOKEN / PINECONE_API_KEY в .env"
            )

        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region

        self.pc = Pinecone(api_key=self.api_key)
        self.index = self._connect()

    def _connect(self):
        """Подключается к существующему индексу или создаёт новый.

        Returns:
            Объект индекса Pinecone для операций с векторами.
        """
        existing = {idx["name"] for idx in self.pc.list_indexes()}
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        return self.pc.Index(self.index_name)

    def upsert(
        self,
        vectors: list[dict[str, Any]],
        namespace: str = "",
    ) -> dict[str, Any]:
        """Записывает (upsert) векторы в индекс.

        Args:
            vectors: Список словарей вида::

                {
                    "id": "vec-1",
                    "values": [0.1, 0.2, ...],
                    "metadata": {"text": "пример"}  # опционально
                }

            namespace: Пространство имён. По умолчанию пустая строка.

        Returns:
            Словарь с результатом операции, например ``{"upserted_count": N}``.
        """
        response = self.index.upsert(vectors=vectors, namespace=namespace)
        return {"upserted_count": response.get("upserted_count", len(vectors))}

    def fetch(
        self,
        ids: list[str],
        namespace: str = "",
    ) -> dict[str, Any]:
        """Читает векторы по их идентификаторам.

        Args:
            ids: Список ID векторов для чтения.
            namespace: Пространство имён. По умолчанию пустая строка.

        Returns:
            Словарь вида ``{id: {"id", "values", "metadata"}}`` для найденных векторов.
        """
        response = self.index.fetch(ids=ids, namespace=namespace)
        vectors = response.get("vectors", {})
        result: dict[str, Any] = {}
        for vector_id, vector_data in vectors.items():
            result[vector_id] = {
                "id": vector_id,
                "values": vector_data.get("values", []),
                "metadata": vector_data.get("metadata") or {},
            }
        return result

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Ищет ближайшие векторы по косинусному (или другой) близости.

        Args:
            vector: Вектор запроса той же размерности, что и индекс.
            top_k: Сколько ближайших результатов вернуть. По умолчанию 5.
            namespace: Пространство имён. По умолчанию пустая строка.
            filter: Опциональный фильтр по метаданным, например
                ``{"category": {"$eq": "food"}}``.
            include_values: Включать ли значения векторов в ответ.
            include_metadata: Включать ли метаданные в ответ.

        Returns:
            Список словарей ``{"id", "score", "values", "metadata"}``,
            отсортированный по убыванию сходства.
        """
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )
        matches = response.get("matches", [])
        return [
            {
                "id": match["id"],
                "score": match.get("score"),
                "values": match.get("values") or [],
                "metadata": match.get("metadata") or {},
            }
            for match in matches
        ]

    def delete(
        self,
        ids: list[str] | None = None,
        namespace: str = "",
        delete_all: bool = False,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Удаляет векторы из индекса.

        Args:
            ids: Список ID для удаления. Не нужен, если ``delete_all=True``
                или задан ``filter``.
            namespace: Пространство имён. По умолчанию пустая строка.
            delete_all: Удалить все векторы в namespace.
            filter: Удалить векторы по фильтру метаданных.

        Returns:
            Словарь со статусом операции, например ``{"status": "ok"}``.
        """
        if not delete_all and not ids and filter is None:
            raise ValueError("Укажите ids, filter или delete_all=True")

        kwargs: dict[str, Any] = {"namespace": namespace}
        if delete_all:
            kwargs["delete_all"] = True
        elif filter is not None:
            kwargs["filter"] = filter
        else:
            kwargs["ids"] = ids

        self.index.delete(**kwargs)
        return {"status": "ok"}

    def stats(self, namespace: str | None = None) -> dict[str, Any]:
        """Возвращает статистику индекса.

        Args:
            namespace: Если задан, возвращает статистику только по этому namespace.

        Returns:
            Словарь со статистикой: ``total_vector_count``, ``dimension``,
            ``namespaces`` и т.д.
        """
        response = self.index.describe_index_stats()
        data = dict(response)
        if namespace is not None:
            namespaces = data.get("namespaces") or {}
            return {
                "namespace": namespace,
                "vector_count": namespaces.get(namespace, {}).get("vector_count", 0),
                "dimension": data.get("dimension"),
            }
        return data


if __name__ == "__main__":
    client = PineconeClient()
    print("Подключено к индексу:", client.index_name)
    print("Статистика:", client.stats())
