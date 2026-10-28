# Векторная база Pinecone + семантический поиск

Домашнее задание по работе с Pinecone: создание индекса, загрузка эмбеддингов, поиск по смыслу и Telegram-бот с долговременной памятью.

## Что сделано

1. **Настройка Pinecone** — индекс `embeddings` (dimension 1536, metric cosine), клиент в `pine.py`, embedding-модель `text-embedding-3-small` через ProxyAPI.
2. **Загрузка данных** — 100 фактов о Солнце и планетах Солнечной системы (NASA NSSDC).
3. **Семантический поиск** — интерактивные и демо-запросы, возврат ID, score и текстов.
4. **Дополнительно** — Telegram-бот с памятью в Pinecone (топ-10 воспоминаний перед каждым ответом, модель `openai/gpt-4o-mini`).

## Структура проекта

| Файл | Назначение |
|------|------------|
| `pine.py` | Клиент Pinecone: connect, upsert, fetch, search, delete |
| `embeddings.py` | Клиент embedding-модели OpenAI (ProxyAPI) |
| `data.py` | 100 фраз о Солнце и планетах |
| `test.py` | Пункт 1: индекс + embedding + тест вектора |
| `load_data.py` | Пункт 2: загрузка 100 векторов в Pinecone |
| `search.py` | Пункт 3: демо-запросы и интерактивный поиск |
| `bot.py` | Telegram-бот с памятью (`SearchVectors`, namespace `memories`) |
| `.env.example` | Шаблон переменных окружения |

## Требования

- Python 3.10+
- Аккаунт [Pinecone](https://www.pinecone.io/)
- Ключ [ProxyAPI](https://console.proxyapi.ru/keys) (OpenAI-совместимый API)
- Для бота — токен от [@BotFather](https://t.me/BotFather)

## Установка

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

Скопируйте `.env.example` → `.env` и заполните ключи:

```env
BOT_TOKEN=...
OPENAI_KEY=...
PINICONE_TOKEN=...
```

## Запуск по пунктам ДЗ

### 1. Индекс + embedding + тест вектора

```bash
python test.py
```

Создаёт/подключает индекс, строит эмбеддинг фразы и пишет тестовый вектор в Pinecone.

### 2. Загрузка ≥20 фраз (здесь — 100)

```bash
python load_data.py
```

Векторизует факты из `data.py` и сохраняет их с ID `solar-000` … `solar-099`.

### 3. Поиск по смыслу

```bash
python search.py
```

Сначала прогоняет 5 демо-запросов, затем принимает ваши запросы в консоли.

Примеры запросов:

- Какая планета самая горячая и почему там сильный парниковый эффект?
- Где самые быстрые ветры и какая планета дальше всех от Солнца?
- Расскажи про кольца и спутники газовых гигантов
- Какая планета красная и какие на ней вулканы и каньоны?
- Из чего состоит Солнце и какая у него температура в ядре?

### 4. Telegram-бот (дополнительно)

```bash
python bot.py
```

Команды бота:

| Команда | Действие |
|---------|----------|
| `/start` | справка |
| `/remember <текст>` | сохранить факт в память |
| `запомни: ...` | то же в свободной форме |
| `/memories <запрос>` | поиск по памяти |
| `/forget_all` | удалить свои воспоминания |

Перед каждым ответом бот вызывает `SearchVectors` (топ-10) и подставляет найденное в контекст `openai/gpt-4o-mini`.

## Технологии

- **Pinecone** — векторное хранилище
- **OpenAI `text-embedding-3-small`** — эмбеддинги (1536)
- **OpenAI `gpt-4o-mini`** — чат бота (через ProxyAPI)
- **aiogram 3** — Telegram-бот
- **python-dotenv** — конфигурация из `.env`

## Источники данных

Факты о планетах основаны на [NASA NSSDC Planetary Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/).

## Безопасность

Файл `.env` с секретами **не коммитится** (см. `.gitignore`). В репозиторий кладётся только `.env.example`.
