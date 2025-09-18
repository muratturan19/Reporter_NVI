# -*- coding: utf-8 -*-
"""
Raporlama Ajanı - NVIDIA Nemotron ile
Windows sistemi için UTF-8 uyumlu
"""

import os
import asyncio
import json
from typing import List, Literal, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Gerekli kütüphaneler
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict
    import httpx
except ImportError as e:
    print(f"Eksik kütüphane: {e}")
    print("Şu komutları çalıştırın:")
    print("pip install langchain-nvidia-ai-endpoints")
    print("pip install langgraph")
    print("pip install httpx")
    print("pip install typing-extensions")

# UTF-8 encoding ayarları
import sys
if sys.platform.startswith('win'):
    # Windows için UTF-8 desteği
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'Turkish_Turkey.UTF-8')
        except locale.Error:
            print("UTF-8 yerel ayarı bulunamadı, varsayılan kullanılacak")

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('report_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env dosyası desteği
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# API Keys - .env dosyasından okunacak
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model ayarları - .env dosyasından (varsayılan değerlerle)
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-nano-9b-v2:free")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "2000"))

# Arama ayarları
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
DEFAULT_SEARCH_QUERIES = int(os.getenv("DEFAULT_SEARCH_QUERIES", "3"))

# Log ayarları
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "report_agent.log")

# Rapor ayarları
REPORT_OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", "raporlar")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "tr")

# API key kontrolü
def check_api_keys():
    """API key'lerini kontrol et"""
    missing_keys = []
    
    if not OPENROUTER_API_KEY:
        missing_keys.append("OPENROUTER_API_KEY")
    if not TAVILY_API_KEY:
        missing_keys.append("TAVILY_API_KEY")
    
    if missing_keys:
        print("❌ Eksik API key'ler:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n📋 .env dosyasını kontrol edin veya environment variable'ları ayarlayın")
        return False
    
    print("✅ Tüm API key'ler mevcut")
    return True

# Çıktı dizinini oluştur
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

# Veri modelleri
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
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

# State tanımları
class AgentState(TypedDict):
    """Ana ajan durumu"""
    topic: str
    report_structure: Optional[str]
    research_data: Optional[str]
    sections: List[Dict[str, Any]]
    final_report: Optional[str]
    messages: List[Any]

class ResearcherState(TypedDict):
    """Araştırmacı ajan durumu"""
    topic: str
    number_of_queries: int
    messages: List[Any]

# NVIDIA Nemotron Model Bağlantısı
def create_llm():
    """NVIDIA Nemotron modelini başlat"""
    try:
        llm = ChatNVIDIA(
            base_url="https://openrouter.ai/api/v1",
            model=MODEL_NAME,
            api_key=OPENROUTER_API_KEY,
            temperature=MODEL_TEMPERATURE,
            max_tokens=MODEL_MAX_TOKENS
        )
        logger.info(f"NVIDIA Nemotron modeli başarıyla yüklendi: {MODEL_NAME}")
        return llm
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        raise

# Tavily Web Search Tool
@tool(parse_docstring=True)
async def search_web(
    queries: List[str],
    topic: Literal["general", "news", "finance"] = "general",
    max_results: int = None
) -> str:
    """Web'de arama yap.

    Args:
        queries (List[str]): Arama sorguları listesi.
        topic (Literal["general", "news", "finance"]): Arama konusu türü.
        max_results (Optional[int]): Maksimum sonuç sayısı. ``None`` ise .env'den alınır.

    Returns:
        str: Arama sonuçları string formatında.
    """
    
    if not TAVILY_API_KEY:
        return "Tavily API key bulunamadı! .env dosyasını kontrol edin."
    
    if max_results is None:
        max_results = SEARCH_MAX_RESULTS
    
    try:
        async with httpx.AsyncClient() as client:
            search_results = []
            
            for query in queries:
                logger.info(f"Arama yapılıyor: {query}")
                
                payload = {
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True,
                    "include_raw_content": False,
                    "max_results": max_results,
                    "topic": topic
                }
                
                response = await client.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    search_results.append({
                        "query": query,
                        "results": result.get("results", []),
                        "answer": result.get("answer", "")
                    })
                else:
                    logger.error(f"Arama hatası: {response.status_code}")
                    search_results.append({
                        "query": query,
                        "error": f"HTTP {response.status_code}"
                    })
        
        # Sonuçları formatla
        formatted_results = "=== ARAŞTIRMA SONUÇLARI ===\n\n"
        for result in search_results:
            formatted_results += f"Sorgu: {result['query']}\n"
            if 'error' in result:
                formatted_results += f"Hata: {result['error']}\n\n"
            else:
                if result.get('answer'):
                    formatted_results += f"Özet: {result['answer']}\n"
                
                for idx, res in enumerate(result.get('results', [])[:3], 1):
                    formatted_results += f"{idx}. {res.get('title', 'Başlık yok')}\n"
                    formatted_results += f"   URL: {res.get('url', 'URL yok')}\n"
                    formatted_results += f"   İçerik: {res.get('content', 'İçerik yok')[:200]}...\n"
                formatted_results += "\n"
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Web arama hatası: {e}")
        return f"Arama sırasında hata oluştu: {str(e)}"

def main():
    """Ana fonksiyon - test amaçlı"""
    print("🤖 Raporlama Ajanı Kurulumu Tamamlandı!")
    print("📁 Dosya: UTF-8 encoding ile kaydedildi")
    print("🔧 API Keys kontrol ediliyor...")
    
    if OPENROUTER_API_KEY:
        print("✅ OpenRouter API Key bulundu")
    else:
        print("❌ OpenRouter API Key bulunamadı")
        
    if TAVILY_API_KEY:
        print("✅ Tavily API Key bulundu")
    else:
        print("❌ Tavily API Key bulunamadı")
    
    print("\n📋 Sonraki adım: Araştırmacı ve yazar ajanlarını oluşturma")

if __name__ == "__main__":
    main()