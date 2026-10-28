"""Пункт 3 ДЗ: семантический поиск по векторам в Pinecone."""

from __future__ import annotations

from embeddings import EmbeddingClient
from pine import PineconeClient

# Демонстрационные запросы для домашнего задания (3–5 уникальных).
DEMO_QUERIES: list[str] = [
    "Какая планета самая горячая и почему там сильный парниковый эффект?",
    "Где самые быстрые ветры и какая планета дальше всех от Солнца?",
    "Расскажи про кольца и спутники газовых гигантов",
    "Какая планета красная и какие на ней вулканы и каньоны?",
    "Из чего состоит Солнце и какая у него температура в ядре?",
]


def run_search(
    query: str,
    pine: PineconeClient,
    embedder: EmbeddingClient,
    top_k: int = 5,
) -> list[dict]:
    """Ищет ближайшие фразы по смыслу запроса.

    Args:
        query: Текстовый запрос пользователя.
        pine: Клиент Pinecone.
        embedder: Клиент embedding-модели.
        top_k: Сколько результатов вернуть.

    Returns:
        Список совпадений с id, score и metadata.
    """
    vector = embedder.embed(query)
    return pine.search(vector=vector, top_k=top_k, include_metadata=True)


def print_results(query: str, matches: list[dict]) -> None:
    """Печатает результаты поиска в читаемом виде."""
    print(f"\nЗапрос: {query}")
    print("-" * 60)
    if not matches:
        print("Ничего не найдено.")
        return
    for i, match in enumerate(matches, start=1):
        meta = match.get("metadata") or {}
        text = meta.get("text", "")
        obj = meta.get("object", "?")
        score = match.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        print(f"{i}. id={match['id']}  score={score_str}  object={obj}")
        print(f"   {text}")
    print()


def demo(pine: PineconeClient, embedder: EmbeddingClient, top_k: int = 3) -> None:
    """Прогоняет 5 уникальных запросов и показывает ответы Pinecone."""
    print("\n=== Демо: 5 смысловых запросов ===")
    for query in DEMO_QUERIES:
        matches = run_search(query, pine, embedder, top_k=top_k)
        print_results(query, matches)


def interactive(pine: PineconeClient, embedder: EmbeddingClient, top_k: int = 5) -> None:
    """Интерактивный режим: пользователь вводит запросы в консоли."""
    print("\n=== Интерактивный поиск ===")
    print("Введите запрос по Солнцу и планетам.")
    print("Команды: Enter пустой / exit / quit — выход; demo — демо-запросы.\n")

    while True:
        try:
            query = input("Запрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not query or query.lower() in {"exit", "quit", "q"}:
            print("Выход.")
            break
        if query.lower() == "demo":
            demo(pine, embedder, top_k=3)
            continue

        matches = run_search(query, pine, embedder, top_k=top_k)
        print_results(query, matches)


def main() -> None:
    pine = PineconeClient(index_name="embeddings", dimension=1536)
    embedder = EmbeddingClient(model="text-embedding-3-small", dimension=1536)
    print(f"Подключено к индексу: {pine.index_name}")

    # Сначала демо для отчёта по ДЗ, затем интерактив.
    demo(pine, embedder, top_k=3)
    interactive(pine, embedder, top_k=5)


if __name__ == "__main__":
    main()
