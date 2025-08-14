from __future__ import annotations

import time
import uuid
from typing import List
import asyncio

from fastapi import FastAPI

from .config import settings
from .models import AskRequest, AskResponse, Source, Usage
from .retriever import Retriever
from .llm import LLM

# Использую structlog, если не удаётся -- фоллбэк на обычный python
try:
    import logging
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logger = structlog.get_logger(__name__)

    def log_info(event: str, **fields):
        logger.info(event, **fields)

except Exception:
    import json, sys, datetime

    def log_info(event: str, **fields):
        record = {"event": event,
                  "ts": datetime.datetime.utcnow().isoformat() + "Z"}
        record.update(fields)
        print(json.dumps(record, ensure_ascii=False))

app = FastAPI(title="Тестовое задание на RAG-сервис")
retriever = Retriever(settings.KB_DIR)
llm = LLM(settings.MODEL_NAME)

# простой ин-мемори кэш
CACHE: dict[str, AskResponse] = {}


@app.get("/health")
async def health():
    return {"status": "ok", "kb_docs": len(retriever.docs),
            "cache_size": len(CACHE)}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Проверка кэша
    if req.question in CACHE:
        cached = CACHE[req.question]
        log_info("ask.cache_hit", request_id=request_id, question=req.question)
        return cached

    # Получаем контекст
    hits = await asyncio.to_thread(retriever.retrieve, req.question,
                                   settings.TOP_K)
    sources: List[Source] = [Source(id=doc_id, score=float(score)) for
                             doc_id, score, _ in hits]

    joined = ""
    for _, _, content in hits:
        if len(joined) >= settings.MAX_CONTEXT_CHARS:
            break
        left = settings.MAX_CONTEXT_CHARS - len(joined)
        joined += "\n\n---\n\n" + content[:left]

    # Генерация в LLM
    gen = await asyncio.to_thread(llm.generate, req.question, joined)
    answer_text = gen["text"]
    usage = gen["usage"]

    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000.0, 2)

    # Логируем метрики
    log_info(
        "ask.metrics",
        request_id=request_id,
        latency_ms=latency_ms,
        model=settings.MODEL_NAME,
        tokens_prompt=usage["prompt_tokens"],
        tokens_completion=usage["completion_tokens"],
        tokens_total=usage["total_tokens"],
        top_k=settings.TOP_K,
        source_ids=[s.id for s in sources],
    )

    resp = AskResponse(
        answer=answer_text,
        sources=sources,
        usage=Usage(**usage),
    )

    # Добавляем в кэш
    CACHE[req.question] = resp

    return resp


