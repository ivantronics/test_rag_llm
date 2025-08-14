from __future__ import annotations
from typing import Dict


# Заглушка, работает оффлайн, генерирует краткий ответ на основе контекста
# Легко подключается внешняя или локальная по этой точке входа
class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, question: str, context: str) -> Dict[str, object]:
        if context.strip():
            answer = (
                    "Краткий ответ: основываясь на базе знаний — " +
                    question.strip() + "\n\n""Главное из контекста:\n" + context.strip()[:500]
            )
        else:
            answer = "Не удалось найти релевантный контекст в базе знаний."

        # Примитивный подсчёт токенов
        prompt_tokens = len((question + context).split())
        completion_tokens = len(answer.split())

        return {
            "text": answer,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
