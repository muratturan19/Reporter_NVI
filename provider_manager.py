"""Provider yönetimi ve dinamik araç oluşturma yardımcıları.

Bu modül, LLM ve arama sağlayıcılarını modüler şekilde yönetmek için
soyut sınıflar ve fabrika yardımcıları sunar. Kullanıcılar runtime'da
farklı sağlayıcı kombinasyonlarını seçebilir ve sistem bunları uygun
LangChain nesnelerine dönüştürür.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Varsayılan maksimum çıktı token sayısı. Ortam değişkeni ile
# özelleştirilebilir ve farklı sağlayıcılar için ortak bir
# başlangıç değeri sağlar.
DEFAULT_MAX_TOKENS = int(os.getenv("REPORTER_DEFAULT_MAX_TOKENS", "4096"))


def _get_max_tokens(*env_names: str, default: Optional[int] = DEFAULT_MAX_TOKENS) -> Optional[int]:
    """Öncelik sırasına göre token limitini ortam değişkenlerinden oku.

    Parametre olarak verilen ortam değişkenleri sırayla kontrol edilir. İlk
    geçerli (tamsayıya dönüştürülebilen) değer döndürülür. Geçersiz değerler
    tespit edildiğinde uyarı loglanır ve sıradaki aday kontrol edilir.

    Args:
        *env_names: Kontrol edilecek ortam değişkenlerinin isimleri.
        default: Hiçbir değer bulunamadığında döndürülecek varsayılan token
            limiti. `None` verilirse limit uygulanmaz.

    Returns:
        Bulunan token limiti ya da default değer.
    """

    for name in env_names:
        if not name:
            continue
        raw_value = os.getenv(name)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except ValueError:
            logger.warning("%s değeri sayıya çevrilemedi: %s", name, raw_value)
    return default


@dataclass
class SearchHit:
    """Tek bir arama sonucunu temsil eder."""

    title: str = "Başlık bulunamadı"
    url: str = ""
    snippet: str = ""


@dataclass
class SearchProviderResult:
    """Arama sağlayıcısından dönen verileri tutar."""

    provider_id: str
    provider_name: str
    query: str
    hits: List[SearchHit] = field(default_factory=list)
    summary: Optional[str] = None
    error: Optional[str] = None


class BaseProvider(ABC):
    """Tüm sağlayıcılar için temel sınıf."""

    provider_type: str = "generic"
    provider_id: str = ""
    display_name: str = ""
    description: str = ""
    strengths: Sequence[str] = ()
    docs_url: Optional[str] = None
    required_env_vars: Sequence[str] = ()
    optional_env_vars: Sequence[str] = ()
    default: bool = False

    def metadata(self) -> Dict[str, Any]:
        """Sağlayıcı hakkında UI'da gösterilecek meta bilgileri döndür."""

        return {
            "id": self.provider_id,
            "name": self.display_name,
            "type": self.provider_type,
            "description": self.description,
            "strengths": list(self.strengths),
            "docs_url": self.docs_url,
            "required_env_vars": list(self.required_env_vars),
            "optional_env_vars": list(self.optional_env_vars),
            "default": self.default,
        }

    def availability_status(self) -> Tuple[bool, Optional[str]]:
        """Sağlayıcının kullanılabilirliğini ve gerekirse hata mesajını döndür."""

        missing = [var for var in self.required_env_vars if not os.getenv(var)]
        if missing:
            message = "Eksik ortam değişkenleri: " + ", ".join(missing)
            return False, message
        return True, None

    def is_available(self) -> bool:
        """Sağlayıcı kullanılabilir mi?"""

        ok, _ = self.availability_status()
        return ok


class BaseLLMProvider(BaseProvider):
    """LLM sağlayıcıları için temel sınıf."""

    provider_type = "llm"

    @abstractmethod
    def create_llm(self):
        """LangChain uyumlu LLM örneği oluştur."""


class BaseSearchProvider(BaseProvider):
    """Arama sağlayıcıları için temel sınıf."""

    provider_type = "search"
    timeout: float = 30.0

    def build_result(
        self,
        query: str,
        *,
        summary: Optional[str] = None,
        hits: Optional[Iterable[Dict[str, Any]] | Iterable[SearchHit]] = None,
        error: Optional[str] = None,
    ) -> SearchProviderResult:
        """Alt sınıflar için yardımcı sonuç oluşturucu."""

        normalized_hits: List[SearchHit] = []

        if hits:
            for item in hits:
                if isinstance(item, SearchHit):
                    normalized_hits.append(item)
                    continue

                if isinstance(item, dict):
                    title = str(item.get("title", "Başlık bulunamadı")).strip() or "Başlık bulunamadı"
                    url = str(item.get("url", "")).strip()
                    snippet = str(item.get("snippet", item.get("content", ""))).strip()
                else:
                    title = str(item)
                    url = ""
                    snippet = ""

                normalized_hits.append(SearchHit(title=title, url=url, snippet=snippet))

        return SearchProviderResult(
            provider_id=self.provider_id,
            provider_name=self.display_name,
            query=query,
            summary=summary,
            hits=normalized_hits,
            error=error,
        )

    async def search(
        self,
        query: str,
        *,
        topic: str = "general",
        max_results: int = 5,
    ) -> SearchProviderResult:
        """Arama yap ve sonuç döndür."""

        raise NotImplementedError


class OpenRouterNemotronProvider(BaseLLMProvider):
    """OpenRouter üzerinden NVIDIA Nemotron modeli."""

    provider_id = "openrouter-nemotron"
    display_name = "OpenRouter · NVIDIA Nemotron"
    description = (
        "OpenRouter API üzerinden NVIDIA Nemotron Nano 9B modelini"
        " kullanır. Uygun maliyetli ve Türkçe performansı dengeli"
        " bir seçenek sunar."
    )
    strengths = (
        "Open-source NVIDIA Nemotron ailesi",
        "OpenRouter kredileri ile erişim",
        "Türkçe ve teknik içerikte stabil sonuçlar",
    )
    docs_url = "https://openrouter.ai/models/nvidia/nemotron-nano-9b-v2"
    required_env_vars = ("OPENROUTER_API_KEY",)
    optional_env_vars = ("MODEL_NAME", "MODEL_TEMPERATURE", "MODEL_MAX_TOKENS")
    default = False

    def availability_status(self) -> Tuple[bool, Optional[str]]:
        ok, reason = super().availability_status()
        if not ok:
            return ok, reason
        try:
            import langchain_nvidia_ai_endpoints  # noqa: F401
        except ImportError:
            return False, "langchain-nvidia-ai-endpoints paketi gerekli"
        return True, None

    def create_llm(self):
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        model = os.getenv("MODEL_NAME", "nvidia/nemotron-nano-9b-v2:free")
        temperature = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
        max_tokens = _get_max_tokens("MODEL_MAX_TOKENS")
        api_key = os.getenv("OPENROUTER_API_KEY")

        logger.info("OpenRouter Nemotron modeli yükleniyor: %s", model)
        return ChatNVIDIA(
            base_url="https://openrouter.ai/api/v1",
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class OpenAIGPT4Provider(BaseLLMProvider):
    """OpenAI GPT-4 ailesi için sağlayıcı."""

    provider_id = "openai-gpt4"
    display_name = "OpenAI · GPT-4o"
    description = (
        "OpenAI'nin GPT-4o ailesini kullanarak yüksek doğrulukta"
        " yanıtlar üretir. Geniş eklenti ekosistemi ve gelişmiş"
        " muhakeme yetenekleri sunar."
    )
    strengths = (
        "Üst düzey muhakeme performansı",
        "Zengin entegrasyon ve eklenti ekosistemi",
        "Çok dilli içerikte güçlü sonuçlar",
    )
    docs_url = "https://platform.openai.com/docs/models"
    required_env_vars = ("OPENAI_API_KEY",)
    optional_env_vars = ("OPENAI_MODEL", "OPENAI_TEMPERATURE", "OPENAI_MAX_TOKENS")

    def availability_status(self) -> Tuple[bool, Optional[str]]:
        ok, reason = super().availability_status()
        if not ok:
            return ok, reason
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            return False, "langchain-openai paketi gerekli"
        return True, None

    def create_llm(self):
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", os.getenv("MODEL_TEMPERATURE", "0.7")))
        max_tokens = _get_max_tokens("OPENAI_MAX_TOKENS", "MODEL_MAX_TOKENS")

        kwargs: Dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
        }

        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        logger.info("OpenAI GPT-4o modeli yükleniyor: %s", model)
        return ChatOpenAI(**kwargs)


class AnthropicClaudeProvider(BaseLLMProvider):
    """Anthropic Claude modeli sağlayıcısı."""

    provider_id = "anthropic-claude"
    display_name = "Anthropic · Claude 3"
    description = (
        "Anthropic Claude 3 ailesi ile hızlı ve güvenli yanıtlar sağlar."
        " Uzun bağlam ve güvenlik filtreleri ile kurumsal ihtiyaçlara"
        " uygun yapı sunar."
    )
    strengths = (
        "Güvenlik odaklı tasarım",
        "Uzun bağlam penceresi",
        "Hızlı ve düşük maliyetli Haiku modeli",
    )
    docs_url = "https://docs.anthropic.com/claude/docs"
    required_env_vars = ("ANTHROPIC_API_KEY",)
    optional_env_vars = ("ANTHROPIC_MODEL", "ANTHROPIC_TEMPERATURE", "ANTHROPIC_MAX_TOKENS")
    default = True

    def availability_status(self) -> Tuple[bool, Optional[str]]:
        ok, reason = super().availability_status()
        if not ok:
            return ok, reason
        try:
            import langchain_anthropic  # noqa: F401
        except ImportError:
            return False, "langchain-anthropic paketi gerekli"
        return True, None

    def create_llm(self):
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        temperature = float(os.getenv("ANTHROPIC_TEMPERATURE", os.getenv("MODEL_TEMPERATURE", "0.7")))
        max_tokens = _get_max_tokens("ANTHROPIC_MAX_TOKENS", "MODEL_MAX_TOKENS")

        kwargs: Dict[str, Any] = {
            "model_name": model,
            "temperature": temperature,
            "api_key": api_key,
        }

        if max_tokens is not None:
            kwargs["max_tokens_to_sample"] = max_tokens

        logger.info("Anthropic Claude modeli yükleniyor: %s", model)
        return ChatAnthropic(**kwargs)


class TavilySearchProvider(BaseSearchProvider):
    """Tavily arama API sağlayıcısı."""

    provider_id = "tavily"
    display_name = "Tavily Search"
    description = "Web araması için optimize edilmiş Tavily API'si"
    strengths = (
        "Otomatik özet üretimi",
        "Hızlı yanıt süresi",
        "Özel arama başlıkları (news, finance) desteği",
    )
    docs_url = "https://docs.tavily.com/"
    required_env_vars = ("TAVILY_API_KEY",)
    optional_env_vars = ("SEARCH_MAX_RESULTS",)
    default = True

    async def search(
        self,
        query: str,
        *,
        topic: str = "general",
        max_results: int = 5,
    ) -> SearchProviderResult:
        available, message = self.availability_status()
        if not available:
            return self.build_result(query, error=message)

        payload = {
            "api_key": os.getenv("TAVILY_API_KEY"),
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": max_results,
            "topic": topic,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post("https://api.tavily.com/search", json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error("Tavily arama hatası", exc_info=exc)
            return self.build_result(query, error=str(exc))

        hits = []
        for item in data.get("results", [])[:max_results]:
            hits.append(
                {
                    "title": item.get("title", "Başlık yok"),
                    "url": item.get("url", ""),
                    "snippet": (item.get("content", "") or "")[:500],
                }
            )

        return self.build_result(query, summary=data.get("answer"), hits=hits)


class ExaSearchProvider(BaseSearchProvider):
    """EXA semantik arama sağlayıcısı."""

    provider_id = "exa"
    display_name = "Exa Semantic Search"
    description = "Semantik web araması ve autoprompt desteği"
    strengths = (
        "Semantik benzerlik tabanlı sonuçlar",
        "Kaynak çeşitliliği",
        "Autoprompt ile daha iyi sorgular",
    )
    docs_url = "https://docs.exa.ai/reference/search"
    required_env_vars = ("EXA_API_KEY",)

    async def search(
        self,
        query: str,
        *,
        topic: str = "general",
        max_results: int = 5,
    ) -> SearchProviderResult:
        available, message = self.availability_status()
        if not available:
            return self.build_result(query, error=message)

        payload = {
            "query": query,
            "useAutoprompt": True,
            "numResults": max_results,
            "type": "neural",
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": os.getenv("EXA_API_KEY", ""),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post("https://api.exa.ai/search", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error("Exa arama hatası", exc_info=exc)
            return self.build_result(query, error=str(exc))

        hits = []
        for item in data.get("results", [])[:max_results]:
            hits.append(
                {
                    "title": item.get("title", "Başlık yok"),
                    "url": item.get("url", ""),
                    "snippet": (item.get("text", "") or "")[:500],
                }
            )

        return self.build_result(query, summary=data.get("summary"), hits=hits)


class SerpAPISearchProvider(BaseSearchProvider):
    """SerpAPI Google arama sağlayıcısı."""

    provider_id = "serpapi"
    display_name = "SerpAPI Google Search"
    description = "Google SERP sonuçlarını sağlayan SerpAPI"
    strengths = (
        "Google sonuçlarına hızlı erişim",
        "Zengin bilgi kartı desteği",
        "Kaynak başına detaylı metadata",
    )
    docs_url = "https://serpapi.com/search-api"
    required_env_vars = ("SERPAPI_API_KEY",)

    async def search(
        self,
        query: str,
        *,
        topic: str = "general",
        max_results: int = 5,
    ) -> SearchProviderResult:
        available, message = self.availability_status()
        if not available:
            return self.build_result(query, error=message)

        params = {
            "engine": "google",
            "q": query,
            "num": max_results,
            "hl": "tr" if topic == "general" else "en",
            "api_key": os.getenv("SERPAPI_API_KEY"),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get("https://serpapi.com/search.json", params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error("SerpAPI arama hatası", exc_info=exc)
            return self.build_result(query, error=str(exc))

        hits = []
        for item in data.get("organic_results", [])[:max_results]:
            hits.append(
                {
                    "title": item.get("title", "Başlık yok"),
                    "url": item.get("link", ""),
                    "snippet": (item.get("snippet", "") or "")[:500],
                }
            )

        summary = None
        answer_box = data.get("answer_box")
        if isinstance(answer_box, dict):
            summary = answer_box.get("answer") or answer_box.get("snippet")

        return self.build_result(query, summary=summary, hits=hits)


class YoucomSearchProvider(BaseSearchProvider):
    """You.com Search API sağlayıcısı."""

    provider_id = "youcom"
    display_name = "You.com Search"
    description = "You.com çok kaynaklı web araması ve kısa özet desteği"
    strengths = (
        "Çoklu kaynaklardan gerçek zamanlı sonuçlar",
        "You.com yanıt motorundan hızlı özetler",
        "Haber ve genel web içeriklerinde geniş kapsama",
    )
    docs_url = "https://docs.you.com/reference/search"
    required_env_vars = ("YOUCOM_API_KEY",)
    optional_env_vars = (
        "YOUCOM_SAFE_SEARCH",
        "YOUCOM_COUNTRY",
        "YOUCOM_LANGUAGE",
        "YOUCOM_DOMAIN",
    )

    api_url = "https://api.you.com/search"

    async def search(
        self,
        query: str,
        *,
        topic: str = "general",
        max_results: int = 5,
    ) -> SearchProviderResult:
        available, message = self.availability_status()
        if not available:
            return self.build_result(query, error=message)

        safe_search = os.getenv("YOUCOM_SAFE_SEARCH") or os.getenv("YOUCOM_SAFESEARCH")
        if not safe_search:
            # Varsayılan değeri orta seviye tut, finans/news için hafif
            safe_search = "Moderate"
        else:
            safe_search = safe_search.capitalize()

        payload: Dict[str, Any] = {
            "query": query,
            "num_web_results": max_results,
            "page": 1,
            "domain": os.getenv("YOUCOM_DOMAIN", "you.com"),
            "safeSearch": safe_search,
        }

        language = os.getenv("YOUCOM_LANGUAGE")
        if not language:
            language = "tr" if topic == "general" else "en"
        payload["language"] = language

        country = os.getenv("YOUCOM_COUNTRY")
        if country:
            payload["country"] = country

        headers = {
            "X-API-Key": os.getenv("YOUCOM_API_KEY", ""),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data: Dict[str, Any]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error("You.com arama hatası", exc_info=exc)
            return self.build_result(query, error=str(exc))

        def _extract_hits(source: Any) -> Iterable[Dict[str, Any]]:
            if isinstance(source, list):
                return source
            if isinstance(source, dict):
                # Bazı yanıtlar results => {"web": [...]}
                for key in ("web", "web_results", "webResults", "results", "searchResults", "hits"):
                    nested = source.get(key)
                    if isinstance(nested, list):
                        return nested
            return []

        hits_source: Iterable[Dict[str, Any]] = []
        for key in (
            "web_results",
            "webResults",
            "searchResults",
            "results",
            "hits",
        ):
            value = data.get(key)
            if value:
                hits_source = _extract_hits(value)
                if hits_source:
                    break

        if not hits_source and isinstance(data, dict):
            hits_source = _extract_hits(data)

        hits: List[Dict[str, Any]] = []
        for item in hits_source:
            if not isinstance(item, dict):
                continue
            hits.append(
                {
                    "title": item.get("title") or item.get("name") or "Başlık yok",
                    "url": item.get("url") or item.get("link") or "",
                    "snippet": item.get("snippet")
                    or item.get("summary")
                    or item.get("description")
                    or "",
                }
            )
            if len(hits) >= max_results:
                break

        summary: Optional[str] = None
        for key in (
            "answer",
            "instant_answer",
            "direct_answer",
            "summary",
            "overview",
        ):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                summary = value.strip()
                break

        if not summary:
            youchat = data.get("youChat") or data.get("youchat")
            if isinstance(youchat, dict):
                summary = youchat.get("response") or youchat.get("message")

        return self.build_result(query, summary=summary, hits=hits)


class ProviderFactory:
    """Sağlayıcı örneklerini yöneten fabrika sınıfı."""

    LLM_PROVIDERS: Dict[str, type[BaseLLMProvider]] = {
        OpenRouterNemotronProvider.provider_id: OpenRouterNemotronProvider,
        OpenAIGPT4Provider.provider_id: OpenAIGPT4Provider,
        AnthropicClaudeProvider.provider_id: AnthropicClaudeProvider,
    }

    SEARCH_PROVIDERS: Dict[str, type[BaseSearchProvider]] = {
        TavilySearchProvider.provider_id: TavilySearchProvider,
        ExaSearchProvider.provider_id: ExaSearchProvider,
        SerpAPISearchProvider.provider_id: SerpAPISearchProvider,
        YoucomSearchProvider.provider_id: YoucomSearchProvider,
    }

    DEFAULT_LLM_PROVIDER_ID = os.getenv(
        "DEFAULT_LLM_PROVIDER", AnthropicClaudeProvider.provider_id
    )
    DEFAULT_SEARCH_PROVIDER_IDS = [
        provider_id
        for provider_id in (os.getenv("DEFAULT_SEARCH_PROVIDERS") or TavilySearchProvider.provider_id).split(",")
        if provider_id.strip()
    ]

    @classmethod
    def normalize_provider_id(cls, provider_id: Optional[str]) -> Optional[str]:
        if provider_id is None:
            return None
        return provider_id.strip().lower() or None

    @classmethod
    def get_llm_provider(cls, provider_id: Optional[str] = None) -> BaseLLMProvider:
        normalized = cls.normalize_provider_id(provider_id) or cls.DEFAULT_LLM_PROVIDER_ID
        provider_cls = cls.LLM_PROVIDERS.get(normalized)
        if not provider_cls:
            raise ValueError(f"Bilinmeyen LLM sağlayıcısı: {provider_id}")
        return provider_cls()

    @classmethod
    def get_search_providers(cls, provider_ids: Optional[Iterable[str]] = None) -> List[BaseSearchProvider]:
        ids: List[str]
        if provider_ids is None:
            ids = cls.DEFAULT_SEARCH_PROVIDER_IDS or [TavilySearchProvider.provider_id]
        else:
            ids = []
            for provider_id in provider_ids:
                normalized = cls.normalize_provider_id(provider_id)
                if normalized and normalized not in ids:
                    ids.append(normalized)

        providers: List[BaseSearchProvider] = []
        for provider_id in ids:
            provider_cls = cls.SEARCH_PROVIDERS.get(provider_id)
            if not provider_cls:
                logger.warning("Bilinmeyen arama sağlayıcısı atlandı: %s", provider_id)
                continue
            providers.append(provider_cls())

        if not providers:
            logger.warning("Geçerli arama sağlayıcısı bulunamadı, Tavily varsayılanı kullanılacak")
            providers.append(TavilySearchProvider())

        return providers

    @classmethod
    def create_llm(cls, provider_id: Optional[str] = None):
        provider = cls.get_llm_provider(provider_id)
        available, message = provider.availability_status()
        if not available:
            raise RuntimeError(f"LLM sağlayıcısı kullanılamıyor: {message}")
        return provider.create_llm()

    @classmethod
    def create_search_tool(
        cls,
        provider_ids: Optional[Iterable[str]] = None,
        *,
        max_results: Optional[int] = None,
    ):
        providers = cls.get_search_providers(provider_ids)
        if max_results is None:
            try:
                default_limit = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
            except ValueError:
                default_limit = 5
        else:
            default_limit = max_results if max_results > 0 else 5

        provider_names = ", ".join(provider.display_name for provider in providers)
        logger.info("Arama sağlayıcıları yapılandırılıyor: %s", provider_names)

        async def _execute_provider(
            provider: BaseSearchProvider,
            query: str,
            topic: str,
            limit: int,
        ) -> SearchProviderResult:
            available, message = provider.availability_status()
            if not available:
                return provider.build_result(query, error=message)

            try:
                return await provider.search(query, topic=topic, max_results=limit)
            except Exception as exc:  # pragma: no cover - hata durumlarını logla
                logger.error(
                    "Arama sağlayıcısı hata verdi: %s", provider.provider_id, exc_info=exc
                )
                return provider.build_result(query, error=str(exc))

        @tool("search_web", parse_docstring=True)
        async def multi_search(
            queries: List[str],
            topic: str = "general",
            max_results: Optional[int] = None,
        ) -> str:
            """Seçili sağlayıcılar ile paralel web araması yap.

            Args:
                queries: Çalıştırılacak arama sorguları listesi.
                topic: Arama bağlamı (general, news, finance).
                max_results: Sağlayıcı başına maksimum sonuç sayısı.

            Returns:
                str: Sağlayıcı bazında gruplanmış arama sonuçları.
            """

            effective_limit = (
                max_results if isinstance(max_results, int) and max_results > 0 else default_limit
            )

            if not isinstance(effective_limit, int) or effective_limit <= 0:
                effective_limit = default_limit

            formatted_output: List[str] = ["=== ARAŞTIRMA SONUÇLARI ===", ""]

            if isinstance(queries, str):
                queries = [queries]

            if not queries:
                return "Arama yapılacak sorgu bulunamadı."

            for query in queries:
                formatted_output.append(f"Sorgu: {query}")

                results = await asyncio.gather(
                    *[
                        _execute_provider(provider, query, topic, effective_limit)
                        for provider in providers
                    ]
                )

                for provider_result in results:
                    formatted_output.append(f"[{provider_result.provider_name}]")

                    if provider_result.error:
                        formatted_output.append(f"⚠️ {provider_result.error}")
                        formatted_output.append("")
                        continue

                    if provider_result.summary:
                        formatted_output.append(f"Özet: {provider_result.summary}")

                    for index, hit in enumerate(provider_result.hits[:effective_limit], 1):
                        formatted_output.append(f"{index}. {hit.title}")
                        if hit.url:
                            formatted_output.append(f"   URL: {hit.url}")
                        snippet = hit.snippet.strip()
                        if snippet:
                            trimmed = snippet[:400] + ("..." if len(snippet) > 400 else "")
                            formatted_output.append(f"   İçerik: {trimmed}")

                    formatted_output.append("")

                formatted_output.append("")

            return "\n".join(formatted_output).strip()

        multi_search.metadata = {
            "provider_ids": [provider.provider_id for provider in providers],
            "provider_names": [provider.display_name for provider in providers],
        }

        return multi_search

    @classmethod
    def get_llm_provider_options(cls) -> List[Dict[str, Any]]:
        options: List[Dict[str, Any]] = []
        for provider_id, provider_cls in cls.LLM_PROVIDERS.items():
            provider = provider_cls()
            available, message = provider.availability_status()
            meta = provider.metadata()
            meta.update({
                "available": available,
                "availability_message": message,
            })
            options.append(meta)

        options.sort(key=lambda item: (not item.get("default", False), item["name"]))
        return options

    @classmethod
    def get_search_provider_options(cls) -> List[Dict[str, Any]]:
        options: List[Dict[str, Any]] = []
        for provider_id, provider_cls in cls.SEARCH_PROVIDERS.items():
            provider = provider_cls()
            available, message = provider.availability_status()
            meta = provider.metadata()
            meta.update({
                "available": available,
                "availability_message": message,
            })
            options.append(meta)

        options.sort(key=lambda item: (not item.get("default", False), item["name"]))
        return options

