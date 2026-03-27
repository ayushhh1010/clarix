"""
LLM wrapper — hybrid approach:
  - Production (APP_ENV=production): Groq API (llama-3.3-70b-versatile)
  - Development (default):           Ollama (llama3.1, runs locally)
"""

import logging
import os
from typing import AsyncGenerator, Union

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_llm: BaseChatModel | None = None


def _get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        if settings.app_env == "production":
            logger.info("LLM: using Groq API (model=%s)", settings.llm_model)
            _llm = ChatGroq(
                model=settings.llm_model,
                api_key=settings.groq_api_key,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        else:
            logger.info("LLM: using Ollama locally (model=%s)", settings.ollama_model)
            _llm = ChatOllama(
                model=settings.ollama_model,
                temperature=settings.llm_temperature,
            )
    return _llm


def reset_llm() -> None:
    """Force re-initialisation (useful in tests or after config change)."""
    global _llm
    _llm = None


def _dict_to_message(msg: dict) -> BaseMessage:
    role = msg["role"]
    content = msg["content"]
    if role == "system":
        return SystemMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:
        return HumanMessage(content=content)


def generate(messages: list[dict]) -> str:
    """Synchronous LLM call — returns the full response text."""
    llm = _get_llm()
    lc_messages = [_dict_to_message(m) for m in messages]
    result = llm.invoke(lc_messages)
    return result.content


async def generate_stream(messages: list[dict]) -> AsyncGenerator[str, None]:
    """Async streaming LLM call — yields response chunks as they arrive."""
    llm = _get_llm()
    lc_messages = [_dict_to_message(m) for m in messages]
    async for chunk in llm.astream(lc_messages):
        if chunk.content:
            yield chunk.content
