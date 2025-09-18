# -*- coding: utf-8 -*-
"""Raporlama ajanı için yapılandırma ve sağlayıcı yönetimi yardımcıları."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from provider_manager import ProviderFactory

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("report_agent.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Ortam değişkenlerini yükle
load_dotenv()

# API key'ler
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

DEFAULT_REQUIRED_KEYS = ("OPENROUTER_API_KEY", "TAVILY_API_KEY")
OPTIONAL_API_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "EXA_API_KEY",
    "SERPAPI_API_KEY",
)

# Genel konfigürasyon
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
DEFAULT_SEARCH_QUERIES = int(os.getenv("DEFAULT_SEARCH_QUERIES", "3"))
REPORT_OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", "raporlar")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "tr")

# Varsayılan sağlayıcılar
DEFAULT_LLM_PROVIDER_ID = ProviderFactory.DEFAULT_LLM_PROVIDER_ID
DEFAULT_SEARCH_PROVIDERS = ProviderFactory.DEFAULT_SEARCH_PROVIDER_IDS

# Çıktı dizinini oluştur
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)


@dataclass
class Section:
    """Rapor bölümü"""

    name: str
    description: str
    research: bool = True
    content: str = ""


@dataclass
class ReportStructure:
    """Rapor yapısı"""

    title: str
    sections: List[Section]
    created_at: datetime = None

    def __post_init__(self) -> None:  # pragma: no cover - basit getter
        if self.created_at is None:
            self.created_at = datetime.now()


def check_api_keys(required_keys: Optional[List[str]] = None) -> bool:
    """Zorunlu ve opsiyonel API key'lerini kontrol et."""

    keys_to_check = required_keys or list(DEFAULT_REQUIRED_KEYS)
    missing_keys = [key for key in keys_to_check if not os.getenv(key)]

    if missing_keys:
        print("❌ Eksik API key'ler:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n📋 .env dosyasını kontrol edin veya environment variable'ları ayarlayın")

        optional_missing = [key for key in OPTIONAL_API_KEYS if not os.getenv(key)]
        if optional_missing:
            print(
                "ℹ️ Opsiyonel sağlayıcılar için eksik anahtarlar: "
                + ", ".join(optional_missing)
            )
        return False

    print("✅ Zorunlu API key'ler hazır")

    optional_missing = [key for key in OPTIONAL_API_KEYS if not os.getenv(key)]
    if optional_missing:
        print("ℹ️ Opsiyonel sağlayıcı anahtarları henüz eklenmemiş: " + ", ".join(optional_missing))
    else:
        print("✅ Tüm opsiyonel API key'ler de mevcut")

    return True


def create_llm(provider_id: Optional[str] = None):
    """Seçili sağlayıcıya göre LLM örneği oluştur."""

    return ProviderFactory.create_llm(provider_id)


def create_search_tool(provider_ids: Optional[List[str]] = None):
    """Seçili sağlayıcıları kullanarak arama aracı oluştur."""

    return ProviderFactory.create_search_tool(provider_ids, max_results=SEARCH_MAX_RESULTS)


# Geriye dönük uyumluluk için varsayılan arama aracı
search_web = create_search_tool()


def get_llm_provider_options() -> List[Dict[str, Any]]:
    """LLM sağlayıcılarının meta bilgilerini döndür."""

    return ProviderFactory.get_llm_provider_options()


def get_search_provider_options() -> List[Dict[str, Any]]:
    """Arama sağlayıcılarının meta bilgilerini döndür."""

    return ProviderFactory.get_search_provider_options()


def main() -> None:
    """Modül doğrudan çalıştırıldığında sağlayıcı durumunu göster."""

    print("🤖 Raporlama Ajanı Kurulumu")
    print("===========================")

    check_api_keys()

    print("\n🔧 Varsayılan sağlayıcılar:")
    print(f"   - LLM: {DEFAULT_LLM_PROVIDER_ID}")
    print(f"   - Arama: {', '.join(DEFAULT_SEARCH_PROVIDERS)}")

    print("\n📚 Desteklenen LLM sağlayıcıları:")
    for option in get_llm_provider_options():
        status = "✅" if option.get("available") else "⚠️"
        print(f"   {status} {option['name']} ({option['id']})")

    print("\n🔍 Desteklenen arama sağlayıcıları:")
    for option in get_search_provider_options():
        status = "✅" if option.get("available") else "⚠️"
        print(f"   {status} {option['name']} ({option['id']})")


if __name__ == "__main__":
    main()
