"""Умный Telegram-бот с долговременной памятью в Pinecone."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from dotenv import load_dotenv
from openai import OpenAI

from embeddings import EmbeddingClient
from pine import PineconeClient

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_KEY")
# Полный id модели работает на универсальном эндпоинте /v1 (не /openai/v1).
CHAT_MODEL = "openai/gpt-4o-mini"
PROXY_BASE_URL = "https://api.proxyapi.ru/v1"
MEMORY_NAMESPACE = "memories"
TOP_K_MEMORIES = 10

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в .env")
if not OPENAI_KEY:
    raise ValueError("OPENAI_KEY не найден в .env")

pine = PineconeClient(index_name="embeddings", dimension=1536)
embedder = EmbeddingClient(model="text-embedding-3-small", dimension=1536)
llm = OpenAI(base_url=PROXY_BASE_URL, api_key=OPENAI_KEY)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

SYSTEM_PROMPT = """Ты умный Telegram-помощник с долговременной памятью.
Используй блок «Воспоминания» как достоверный контекст о пользователе и прошлых фактах.
Если воспоминания релевантны вопросу — опирайся на них.
Если данных недостаточно — скажи об этом честно и задай уточняющий вопрос.
Отвечай кратко, по делу и на языке пользователя.
Когда пользователь сообщает полезный факт о себе, предпочтениях, планах или важных данных —
подтверди, что учтёшь это (система сохранит факт отдельно)."""


def SearchVectors(
    query: str,
    top_k: int = TOP_K_MEMORIES,
    user_id: int | None = None,
) -> list[dict[str, Any]]:
    """Ищет топ-K подходящих воспоминаний в Pinecone.

    Args:
        query: Текст запроса для семантического поиска.
        top_k: Сколько воспоминаний вернуть. По умолчанию 10.
        user_id: Если задан, ищет только воспоминания этого пользователя.

    Returns:
        Список словарей ``{"id", "score", "text", "metadata"}``.
    """
    vector = embedder.embed(query)
    filters: dict[str, Any] | None = None
    if user_id is not None:
        filters = {"user_id": {"$eq": str(user_id)}}

    matches = pine.search(
        vector=vector,
        top_k=top_k,
        namespace=MEMORY_NAMESPACE,
        filter=filters,
        include_metadata=True,
    )
    results: list[dict[str, Any]] = []
    for match in matches:
        meta = match.get("metadata") or {}
        results.append(
            {
                "id": match["id"],
                "score": match.get("score"),
                "text": meta.get("text", ""),
                "metadata": meta,
            }
        )
    return results


def SaveMemory(
    text: str,
    user_id: int,
    source: str = "user",
) -> dict[str, Any]:
    """Сохраняет полезное воспоминание в Pinecone.

    Args:
        text: Текст факта/воспоминания.
        user_id: Telegram ID пользователя.
        source: Источник записи (``user``, ``auto``, ``command``).

    Returns:
        Словарь ``{"id", "status"}`` с ID сохранённого вектора.
    """
    text = text.strip()
    if not text:
        raise ValueError("Пустой текст нельзя сохранить в память")

    values = embedder.embed(text)
    memory_id = f"mem-{user_id}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    pine.upsert(
        [
            {
                "id": memory_id,
                "values": values,
                "metadata": {
                    "text": text,
                    "user_id": str(user_id),
                    "source": source,
                    "topic": "memory",
                    "created_at": int(time.time()),
                },
            }
        ],
        namespace=MEMORY_NAMESPACE,
    )
    return {"id": memory_id, "status": "ok"}


def format_memories(memories: list[dict[str, Any]]) -> str:
    """Собирает блок воспоминаний для системного контекста."""
    if not memories:
        return "Воспоминания: пока пусто."
    lines = ["Воспоминания (топ релевантных):"]
    for i, item in enumerate(memories, start=1):
        score = item.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
        lines.append(f"{i}. [{score_str}] {item.get('text', '')}")
    return "\n".join(lines)


def chat_with_memory(user_text: str, memories: list[dict[str, Any]]) -> str:
    """Генерирует ответ модели с учётом найденных воспоминаний.

    Args:
        user_text: Сообщение пользователя.
        memories: Топ воспоминаний из SearchVectors.

    Returns:
        Текст ответа ассистента.
    """
    context = format_memories(memories)
    response = llm.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context},
            {"role": "user", "content": user_text},
        ],
        temperature=0.4,
    )
    return (response.choices[0].message.content or "").strip()


def extract_useful_facts(user_text: str) -> list[str]:
    """Извлекает из сообщения пользователя факты, которые стоит запомнить.

    Args:
        user_text: Исходное сообщение.

    Returns:
        Список коротких фактов; пустой список, если запоминать нечего.
    """
    prompt = (
        "Из сообщения пользователя извлеки полезные факты для долговременной памяти "
        "(имя, предпочтения, планы, даты, важные данные). "
        "Верни ТОЛЬКО JSON-массив строк. Если фактов нет — верни [].\n\n"
        f"Сообщение: {user_text}"
    )
    response = llm.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Ты модуль извлечения фактов. Отвечай только валидным JSON-массивом строк.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        # На случай обёртки ```json ... ```
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        return []
    return []


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """Приветствие и краткая справка."""
    await message.answer(
        "Привет! Я умный помощник с памятью в Pinecone.\n\n"
        "Просто пиши мне — перед ответом я подтяну топ-10 подходящих воспоминаний.\n"
        "Полезные факты сохраняю сам, либо явно через:\n"
        "• /remember <текст>\n"
        "• или «запомни: ...»\n"
        "• /memories <запрос> — поиск по памяти\n"
        "• /forget_all — очистить твои воспоминания"
    )


@dp.message(Command("remember"))
async def cmd_remember(message: Message, command: CommandObject) -> None:
    """Явно сохраняет факт в память пользователя."""
    text = (command.args or "").strip()
    if not text:
        await message.answer("Использование: /remember Я живу в Казани и люблю астрономию")
        return
    user_id = message.from_user.id if message.from_user else 0
    saved = await asyncio.to_thread(SaveMemory, text, user_id, "command")
    await message.answer(f"Запомнил ✅\nID: `{saved['id']}`", parse_mode="Markdown")


@dp.message(Command("memories"))
async def cmd_memories(message: Message, command: CommandObject) -> None:
    """Показывает топ воспоминаний по запросу через SearchVectors."""
    query = (command.args or "").strip() or (message.text or "обо мне")
    # Убираем саму команду из текста, если args пустые
    if query.startswith("/memories"):
        query = "обо мне"
    user_id = message.from_user.id if message.from_user else 0
    memories = await asyncio.to_thread(SearchVectors, query, TOP_K_MEMORIES, user_id)
    if not memories:
        await message.answer("Пока нет подходящих воспоминаний.")
        return
    lines = ["Топ воспоминаний:"]
    for i, item in enumerate(memories, start=1):
        score = item.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
        lines.append(f"{i}. `{item['id']}` ({score_str})\n{item['text']}")
    await message.answer("\n\n".join(lines), parse_mode="Markdown")


@dp.message(Command("forget_all"))
async def cmd_forget_all(message: Message) -> None:
    """Удаляет все воспоминания текущего пользователя."""
    user_id = message.from_user.id if message.from_user else 0
    def _delete_user_memories() -> dict[str, Any]:
        return pine.delete(
            namespace=MEMORY_NAMESPACE,
            filter={"user_id": {"$eq": str(user_id)}},
        )

    await asyncio.to_thread(_delete_user_memories)
    await message.answer("Все твои воспоминания удалены.")


@dp.message(F.text)
async def on_text(message: Message) -> None:
    """Основной диалог: SearchVectors → ответ модели → автосохранение фактов."""
    user_text = (message.text or "").strip()
    if not user_text:
        return

    user_id = message.from_user.id if message.from_user else 0
    lower = user_text.lower()

    # Явная команда «запомни: ...»
    if lower.startswith("запомни:") or lower.startswith("запомни "):
        fact = user_text.split(":", 1)[-1].strip() if ":" in user_text else user_text[7:].strip()
        if not fact:
            await message.answer("Напиши факт после «запомни:».")
            return
        saved = await asyncio.to_thread(SaveMemory, fact, user_id, "user")
        await message.answer(f"Запомнил ✅\nID: `{saved['id']}`", parse_mode="Markdown")
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    memories = await asyncio.to_thread(SearchVectors, user_text, TOP_K_MEMORIES, user_id)
    answer = await asyncio.to_thread(chat_with_memory, user_text, memories)
    await message.answer(answer)

    # Автосохранение полезных фактов из реплики пользователя
    facts = await asyncio.to_thread(extract_useful_facts, user_text)
    for fact in facts[:3]:
        await asyncio.to_thread(SaveMemory, fact, user_id, "auto")


async def main() -> None:
    """Запускает long-polling бота."""
    print("Бот запущен. Модель:", CHAT_MODEL)
    print("Память: namespace =", MEMORY_NAMESPACE)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
