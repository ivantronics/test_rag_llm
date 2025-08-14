# Архитектура (кратко)

```
Клеинт > /ask (FastAPI)
   Retriever (TF‑IDF по dataset/kb/*.md)
   Контекст = топ‑k документов (конкатенация, обрезка по MAX_CONTEXT_CHARS)
   LLM.generate(question, context) = ответ + usage
   Логирование метрик (request_id, latency_ms, модель, токены, top‑k, источники)
```

## Почему так
- TF‑IDF (sklearn) -- быстрый и надёжный старт без тяжёлых зависимостей, в коде есть фолбек на чистый Python (подробнее в readme).
- Простая схема RAG соответствует поставленному заданию, контекст собирается из топ‑k документов.
- LLM -- заглушка по умолчанию (подробнее в readme) легко заменить на реальную модель.
- Воспроизводимость: Docker + docker compose + CI (lint, tests, docker build).

