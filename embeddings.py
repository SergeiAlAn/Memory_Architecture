"""Клиент для получения эмбеддингов через OpenAI API (ProxyAPI)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class EmbeddingClient:
    """Клиент embedding-модели OpenAI через ProxyAPI.

    Attributes:
        model: Имя модели эмбеддингов.
        dimension: Ожидаемая размерность вектора.
        client: Экземпляр OpenAI-клиента.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: str | None = None,
        base_url: str = "https://api.proxyapi.ru/openai/v1",
    ) -> None:
        """Инициализирует клиент embedding-модели.

        Args:
            model: Модель эмбеддингов. По умолчанию ``text-embedding-3-small``.
            dimension: Ожидаемая длина вектора. Для small-модели — 1536.
            api_key: Ключ API. Если не передан, читается из ``OPENAI_KEY``.
            base_url: Базовый URL ProxyAPI.

        Returns:
            None.
        """
        self.model = model
        self.dimension = dimension
        key = api_key or os.getenv("OPENAI_KEY")
        if not key:
            raise ValueError("API-ключ не найден. Укажите api_key или OPENAI_KEY в .env")

        self.client = OpenAI(base_url=base_url, api_key=key)

    def embed(self, text: str) -> list[float]:
        """Преобразует один текст в вектор эмбеддинга.

        Args:
            text: Входная строка для векторизации.

        Returns:
            Список float длины ``self.dimension``.
        """
        response = self.client.embeddings.create(input=text, model=self.model)
        vector = response.data[0].embedding
        if len(vector) != self.dimension:
            raise ValueError(
                f"Ожидалась размерность {self.dimension}, получено {len(vector)}"
            )
        return vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Преобразует список текстов в список векторов.

        Args:
            texts: Список строк для векторизации.

        Returns:
            Список векторов той же длины, что и ``texts``.
        """
        if not texts:
            return []
        response = self.client.embeddings.create(input=texts, model=self.model)
        # API сохраняет порядок input → data[i].index
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Ожидалась размерность {self.dimension}, получено {len(vector)}"
                )
        return vectors
