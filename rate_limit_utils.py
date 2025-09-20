"""Yedek sağlayıcı yönetimi için rate limit yardımcıları."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


RATE_LIMIT_KEYWORDS = (
    "rate limit",
    "too many requests",
    "429",
    "exceeded your",
    "exceed your",
)


def is_rate_limit_exception(exc: BaseException) -> bool:
    """Bir hatanın rate limit kaynaklı olup olmadığını tespit et."""

    if isinstance(exc, httpx.HTTPStatusError):
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 429:
            return True

    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True

    name = exc.__class__.__name__.lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True

    message = str(exc).lower()
    return any(keyword in message for keyword in RATE_LIMIT_KEYWORDS)


@dataclass
class ProviderRateLimitError(Exception):
    """Sağlayıcı bazlı rate limit hatası."""

    provider_type: str
    provider_id: str
    original_exception: BaseException

    def __str__(self) -> str:  # pragma: no cover - hata mesajı oluşturucu
        provider_info = f"{self.provider_type}:{self.provider_id}"
        return f"Rate limit ({provider_info}): {self.original_exception}"  # pragma: no cover


class RateLimitAwareLLM:
    """LLM çağrılarını rate limit hatalarına karşı sarmalayan yardımcı."""

    def __init__(self, llm: Any, provider_id: str):
        self._llm = llm
        self.provider_id = provider_id

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegasyon
        return getattr(self._llm, item)

    async def ainvoke(self, *args, **kwargs):
        try:
            return await self._llm.ainvoke(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit_exception(exc):
                raise ProviderRateLimitError("llm", self.provider_id, exc) from exc
            raise

    def bind_tools(self, tools):
        bound = self._llm.bind_tools(tools)
        return RateLimitAwareLLMBinding(bound, self.provider_id)


class RateLimitAwareLLMBinding:
    """Araç bağlı LLM nesnesi için rate limit sarmalayıcı."""

    def __init__(self, binding: Any, provider_id: str):
        self._binding = binding
        self.provider_id = provider_id

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegasyon
        return getattr(self._binding, item)

    async def ainvoke(self, *args, **kwargs):
        try:
            return await self._binding.ainvoke(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit_exception(exc):
                raise ProviderRateLimitError("llm", self.provider_id, exc) from exc
            raise

