from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List


class AskRequest(BaseModel):
    question: str = Field(..., description="Вопрос пользователя")


class Source(BaseModel):
    id: str
    score: float


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    usage: Usage
