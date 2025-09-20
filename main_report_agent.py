# -*- coding: utf-8 -*-
"""
Ana Rapor Ajanı - Tüm bileşenleri birleştiren tam sistem
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Iterable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import logging

# Diğer modüllerden import
from provider_manager import ProviderFactory
from report_agent_setup import (
    Section,
    ReportStructure,
    DEFAULT_LLM_PROVIDER_ID,
    DEFAULT_SEARCH_PROVIDERS,
)
from researcher_agent import ResearcherAgent
from writer_agent import WriterAgent, ReportCompiler
from quality_control_agent import ReportQualityAgent
from rate_limit_utils import ProviderRateLimitError

logger = logging.getLogger(__name__)


def _merge_unique_messages(base: List[str], additional: Iterable[Any]) -> List[str]:
    """Mesaj listelerini tekrar etmeyecek şekilde birleştir."""

    merged = list(base)
    for item in additional:
        if item is None:
            continue
        text = str(item).strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def _normalize_plan_response(data: Any) -> Any:
    """Temiz JSON anahtarları elde etmek için plan çıktısını normalize et."""

    if isinstance(data, dict):
        normalized: Dict[str, Any] = {}
        for key, value in data.items():
            new_key = key
            if isinstance(key, str):
                new_key = key.strip().strip('"').strip("'")
                new_key = new_key.lower()
            normalized[new_key] = _normalize_plan_response(value)
        return normalized
    if isinstance(data, list):
        return [_normalize_plan_response(item) for item in data]
    if isinstance(data, str):
        text = data.strip()

        # Kod bloğu işaretlerini temizle (```json ... ```)
        if text.startswith("```") and text.endswith("```"):
            inner = text[3:-3].strip()
            first_newline = inner.find("\n")
            if first_newline != -1:
                first_line = inner[:first_newline].strip().lower()
                if first_line in {"json", "jsonc"}:
                    inner = inner[first_newline + 1 :]
            text = inner.strip()

        # JSON olarak tekrar parse etmeyi dene (ör. çift tırnaklı string çıktısı)
        if (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        ):
            try:
                reparsed = json.loads(text)
            except json.JSONDecodeError:
                return text
            else:
                return _normalize_plan_response(reparsed)

        return text
    return data


def _normalize_key_name(key: Any) -> Optional[str]:
    """Anahtar isimlerini güvenli şekilde normalize et."""

    if not isinstance(key, str):
        return None
    return key.strip().strip('"').strip("'").lower()


def _get_first_value(data: Any, keys: List[str]) -> Optional[Any]:
    """Verilen anahtar listesinden ilk mevcut değeri döndür."""

    if isinstance(data, dict):
        normalized_map: Dict[str, Any] = {}
        for raw_key, value in data.items():
            normalized_key = _normalize_key_name(raw_key)
            if normalized_key is None:
                continue
            if normalized_key not in normalized_map:
                normalized_map[normalized_key] = value

        for key in keys:
            lookup_key = _normalize_key_name(key)
            if not lookup_key:
                continue
            value = normalized_map.get(lookup_key)
            if value not in (None, ""):
                return value
    elif isinstance(data, list):
        for item in data:
            value = _get_first_value(item, keys)
            if value not in (None, ""):
                return value
    return None


def _coerce_bool(value: Any) -> bool:
    """LLM çıktısından gelen bool değerini normalize et."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"true", "evet", "yes", "1", "doğru"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _parse_section_entry(entry: Any) -> Optional[Section]:
    """Modelden gelen bölüm verisini Section nesnesine dönüştür."""

    if isinstance(entry, dict):
        section_data = _normalize_plan_response(entry)
        name = _get_first_value(section_data, ["name", "title", "başlık", "isim"])
        description = _get_first_value(
            section_data,
            ["description", "açıklama", "aciklama", "summary", "özeti", "özet"],
        )
        research_flag = section_data.get("research")
        if research_flag is None:
            research_flag = section_data.get("araştırma")
        research = _coerce_bool(research_flag)
    elif isinstance(entry, str):
        text = entry.strip()
        if not text:
            return None
        if (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        ):
            try:
                parsed_entry = json.loads(text)
            except json.JSONDecodeError:
                parsed_entry = None
            if parsed_entry is not None:
                return _parse_section_entry(parsed_entry)
        if ":" in text:
            name, desc = text.split(":", 1)
        elif " - " in text:
            name, desc = text.split(" - ", 1)
        else:
            name, desc = text, ""
        name = name.strip()
        description = desc.strip()
        research = False
    else:
        return None

    if not name:
        return None

    if not description:
        description = "Bu bölüm için açıklama sağlanmadı."

    return Section(name=name, description=description, research=research)


# Planlama promptı
REPORT_PLANNER_PROMPT = """Sen bir rapor planlama uzmanısın. Konu hakkında araştırma raporu yapısı oluşturacaksın.

SADECE JSON formatında yanıt ver. Başka hiçbir metin ekleme.

JSON formatı:
{{
  "title": "Rapor başlığı burada",
  "sections": [
    {{"name": "Bölüm adı", "description": "Bu bölümün içeriği", "research": true}},
    {{"name": "Başka bölüm", "description": "Bu bölümün içeriği", "research": false}}
  ]
}}

4-6 bölümlü bir rapor planla. research: true olanlar için ek araştırma yapılacak."""

PLANNER_HUMAN_PROMPT = """
Konu: {topic}

Araştırma Verileri:
{research_data}

Bu bilgileri kullanarak 4-6 bölümden oluşan kapsamlı bir rapor yapısı oluştur. JSON formatında yanıtla.
"""


@dataclass
class ReportAgentState:
    """Ana rapor ajan durumu"""

    topic: str = ""
    research_data: str = ""
    report_structure: Optional[ReportStructure] = None
    sections_content: List[str] = field(default_factory=list)
    final_report: str = ""
    messages: List[Any] = field(default_factory=list)
    quality_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportRunResult:
    """Rapor çalıştırma sonucunu ve meta bilgilerini temsil eder."""

    content: str = ""
    error: Optional[str] = None
    fallback_messages: List[str] = field(default_factory=list)
    llm_provider_id: str = ""
    search_provider_ids: List[str] = field(default_factory=list)


class MainReportAgent:
    """Ana Rapor Ajan Sınıfı - Sağlayıcı seçimleri ile tüm sistemi yönetir"""

    def __init__(
        self,
        llm_provider_id: Optional[str] = None,
        search_provider_ids: Optional[List[str]] = None,
    ):
        # Sağlayıcı seçimlerini kaydet
        self.llm_provider_id = llm_provider_id or DEFAULT_LLM_PROVIDER_ID
        self.search_provider_ids = (
            list(search_provider_ids)
            if search_provider_ids
            else list(DEFAULT_SEARCH_PROVIDERS)
        )

        # Model ve araçları başlat
        self.llm = ProviderFactory.create_llm(self.llm_provider_id)
        self.search_tool = ProviderFactory.create_search_tool(self.search_provider_ids)

        logger.info(
            "LLM sağlayıcısı: %s | Arama sağlayıcıları: %s",
            self.llm_provider_id,
            ", ".join(self.search_provider_ids),
        )

        # Alt ajanları başlat
        self.researcher = ResearcherAgent(self.llm, self.search_tool)
        self.writer = WriterAgent(self.llm, self.search_tool)
        self.compiler = ReportCompiler(self.llm)
        self.quality_agent = ReportQualityAgent(self.llm)

        # Planlama promptu
        self.planner_prompt = ChatPromptTemplate.from_messages([
            ("system", REPORT_PLANNER_PROMPT),
            ("human", PLANNER_HUMAN_PROMPT)
        ])

        # Ana grafı oluştur
        self.graph = self._build_main_graph()

    def _reset_search_metadata(self) -> None:
        """Arama aracı meta verilerini temizle."""

        if hasattr(self.search_tool, "fallback_events"):
            try:
                self.search_tool.fallback_events.clear()
            except Exception:  # pragma: no cover - savunma amaçlı
                self.search_tool.fallback_events = []
        if hasattr(self.search_tool, "last_metadata"):
            self.search_tool.last_metadata = {}

    def _gather_search_feedback(self) -> Tuple[List[str], List[str]]:
        """Arama aracından fallback bilgilerini topla."""

        metadata = getattr(self.search_tool, "last_metadata", {}) or {}
        events = metadata.get("fallbacks")
        if not events:
            events = getattr(self.search_tool, "fallback_events", [])

        fallback_messages: List[str] = [
            str(item).strip()
            for item in events or []
            if str(item).strip()
        ]

        used_providers = metadata.get("used_providers")
        if isinstance(used_providers, list) and used_providers:
            normalized: List[str] = []
            for provider_id in used_providers:
                pid = str(provider_id).strip()
                if pid and pid not in normalized:
                    normalized.append(pid)
            search_ids = normalized
        else:
            search_ids = list(self.search_provider_ids)

        return fallback_messages, search_ids

    def _build_main_graph(self):
        """Ana rapor oluşturma grafını oluştur"""

        async def initial_research(state: ReportAgentState):
            """İlk araştırma"""
            logger.info("İlk araştırma başlatılıyor...")

            research_result = await self.researcher.research(
                topic=state.topic,
                number_of_queries=4
            )

            # Araştırma verilerini topla
            research_content = ""
            for message in research_result["messages"]:
                if hasattr(message, 'content') and message.content:
                    if "ARAŞTIRMA SONUÇLARI" in str(message.content):
                        research_content += str(message.content) + "\n\n"

            state.research_data = research_content
            state.messages.extend(research_result["messages"])

            logger.info("İlk araştırma tamamlandı")
            return state

        async def plan_report(state: ReportAgentState):
            """Rapor yapısını planla"""
            logger.info("Rapor planlanıyor...")

            topic = getattr(state, "topic", "")
            research_data = getattr(state, "research_data", "")
            if not isinstance(research_data, str):
                research_data = str(research_data)

            messages = self.planner_prompt.format_messages(
                topic=topic,
                research_data=research_data[:2000]  # İlk 2000 karakter
            )

            response = await self.llm.ainvoke(messages)

            from json_parser_fix import parse_json_from_response, create_fallback_structure

            try:
                plan_data = parse_json_from_response(response.content)

                if not isinstance(plan_data, dict):
                    raise TypeError("Model yanıtı dict formatında değil")
                if "title" not in plan_data or "sections" not in plan_data:
                    raise KeyError("JSON'da gerekli alanlar bulunamadı")

                sections_data = plan_data.get("sections", [])
                if isinstance(sections_data, dict):
                    sections_iterable = list(sections_data.values())
                elif isinstance(sections_data, list):
                    sections_iterable = sections_data
                else:
                    raise TypeError("sections alanı list ya da dict değil")

                sections: List[Section] = []
                for section_data in sections_iterable:
                    if not isinstance(section_data, dict):
                        logger.warning("Geçersiz bölüm verisi atlandı: %s", section_data)
                        continue
                    sections.append(Section(
                        name=section_data.get("name", "Adsız Bölüm"),
                        description=section_data.get("description", "Açıklama yok"),
                        research=_coerce_bool(section_data.get("research", False))
                    ))

                if not sections:
                    raise ValueError("Plan çıktısında kullanılabilir bölüm bulunamadı")

                report_structure = ReportStructure(
                    title=plan_data.get("title", f"{topic or 'Genel'} Raporu"),
                    sections=sections
                )

                logger.info(
                    "Rapor planlandı: %s bölüm - %s",
                    len(sections),
                    report_structure.title
                )

            except Exception as e:
                logger.error(f"JSON parsing hatası: {e}")
                logger.warning("Fallback yapı kullanılıyor")

                fallback_topic = topic or "Genel Konu"
                plan_data = create_fallback_structure(fallback_topic)

                sections: List[Section] = []
                for section_data in plan_data["sections"]:
                    sections.append(Section(
                        name=section_data["name"],
                        description=section_data["description"],
                        research=_coerce_bool(section_data["research"])
                    ))

                report_structure = ReportStructure(
                    title=plan_data["title"],
                    sections=sections
                )
                logger.info(f"Fallback rapor yapısı oluşturuldu: {len(sections)} bölüm")

            state.report_structure = report_structure
            return state

        async def write_sections(state: ReportAgentState):
            """Tüm bölümleri yaz"""
            logger.info("Bölümler yazılıyor...")

            if not state.report_structure:
                logger.error("Rapor yapısı bulunamadı!")
                return state

            sections_content: List[str] = []

            for i, section in enumerate(state.report_structure.sections, 1):
                logger.info(f"Bölüm yazılıyor: {section.name}")

                content = await self.writer.write_section(
                    section_name=section.name,
                    section_description=section.description,
                    section_index=i,
                    research_data=state.research_data,
                    needs_research=section.research
                )

                sections_content.append(content)

                # Kısa bekleme (rate limiting için)
                await asyncio.sleep(1)

            state.sections_content = sections_content
            logger.info(f"Tüm bölümler yazıldı: {len(sections_content)} bölüm")

            return state

        async def compile_final_report(state: ReportAgentState):
            """Final raporu derle"""
            logger.info("Final rapor derleniyor...")

            final_report = await self.compiler.compile_report(
                topic=state.topic,
                sections=state.sections_content
            )

            state.final_report = final_report
            logger.info("Final rapor derlendi")

            return state

        async def quality_control(state: ReportAgentState):
            """Rapor kalite kontrolü ve düzeltme"""
            logger.info("Kalite kontrolü başlatılıyor...")

            if not state.final_report:
                logger.warning("Final rapor bulunamadı, kalite kontrolü atlanıyor")
                return state

            try:
                # Kalite analizi ve düzeltme
                fixed_report, quality_analysis = await self.quality_agent.process_report(state.final_report)

                # Kalite skorunu logla
                score = quality_analysis.get("overall_score", "unknown")
                severity = quality_analysis.get("severity", "unknown")

                logger.info(f"Kalite skoru: {score}, Önem derecesi: {severity}")

                # Düzeltilmiş raporu kaydet
                state.final_report = fixed_report

                # Quality metadata'yı state'e ekle
                state.quality_metadata = quality_analysis

                logger.info("Kalite kontrolü tamamlandı")

            except Exception as e:
                logger.error(f"Kalite kontrolü hata verdi: {e}")
                # Hata durumunda orijinal raporu koru
                logger.warning("Orijinal rapor korundu")

            return state

        # Graf oluştur
        workflow = StateGraph(ReportAgentState)

        # Düğümler
        workflow.add_node("research", initial_research)
        workflow.add_node("plan", plan_report)
        workflow.add_node("write", write_sections)
        workflow.add_node("compile", compile_final_report)
        workflow.add_node("quality", quality_control)

        # Kenarlar
        workflow.add_edge(START, "research")
        workflow.add_edge("research", "plan")
        workflow.add_edge("plan", "write")
        workflow.add_edge("write", "compile")
        workflow.add_edge("compile", "quality")
        workflow.add_edge("quality", END)

        return workflow.compile()

    async def test_planning(self, topic: str):
        """Sadece planlama aşamasını test et"""
        messages = self.planner_prompt.format_messages(
            topic=topic,
            research_data="Test araştırma verisi"
        )

        response = await self.llm.ainvoke(messages)
        print("Model yanıtı:")
        print(response.content)
        print("-" * 50)

        # JSON parsing test
        try:
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            json_str = response.content[json_start:json_end]

            print("Çıkarılan JSON:")
            print(json_str)
            print("-" * 50)

            plan_data = json.loads(json_str)
            print("Parse edilmiş data:")
            print(json.dumps(plan_data, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"Hata: {e}")

    async def _handle_rate_limit_error(
        self,
        exc: ProviderRateLimitError,
        topic: str,
        fallback_messages: List[str],
        attempted_llms: List[str],
    ) -> ReportRunResult:
        """Rate limit hatası sonrası fallback stratejisini uygula."""

        search_fallbacks, used_search_ids = self._gather_search_feedback()
        fallback_messages = _merge_unique_messages(fallback_messages, search_fallbacks)

        if exc.provider_type == "llm":
            fallback_id = ProviderFactory.find_alternative_llm_provider(
                exclude=attempted_llms
            )
            current_name = ProviderFactory.get_llm_display_name(self.llm_provider_id)

            if fallback_id:
                fallback_name = ProviderFactory.get_llm_display_name(fallback_id)
                switch_message = (
                    f"{current_name} sağlayıcısı rate limit nedeniyle beklemeye alındı. "
                    f"{fallback_name} ile devam ediliyor."
                )
                fallback_messages = _merge_unique_messages(
                    fallback_messages, [switch_message]
                )

                fallback_agent = MainReportAgent(
                    llm_provider_id=fallback_id,
                    search_provider_ids=self.search_provider_ids,
                )

                fallback_result = await fallback_agent.generate_report(
                    topic,
                    attempted_llms=attempted_llms,
                )

                fallback_result.fallback_messages = _merge_unique_messages(
                    fallback_messages,
                    fallback_result.fallback_messages,
                )
                return fallback_result

            detail = str(exc.original_exception).strip()
            message = (
                f"{current_name} sağlayıcısı rate limit nedeniyle kullanılamadı ve uygun alternatif bulunamadı."
            )
            if detail and detail not in message:
                message = f"{message}\nDetay: {detail}"

            fallback_messages = _merge_unique_messages(fallback_messages, [message])
            return ReportRunResult(
                content="",
                error=message,
                fallback_messages=fallback_messages,
                llm_provider_id=self.llm_provider_id,
                search_provider_ids=used_search_ids,
            )

        provider_name = ProviderFactory.get_search_display_name(exc.provider_id)
        detail = str(exc.original_exception).strip()
        message = f"{provider_name} sağlayıcısı rate limit nedeniyle kullanılamadı."
        if detail and detail not in message:
            message = f"{message}\nDetay: {detail}"

        fallback_messages = _merge_unique_messages(fallback_messages, [message])
        return ReportRunResult(
            content="",
            error=message,
            fallback_messages=fallback_messages,
            llm_provider_id=self.llm_provider_id,
            search_provider_ids=used_search_ids,
        )

    async def generate_report(
        self,
        topic: str,
        *,
        attempted_llms: Optional[List[str]] = None,
    ) -> ReportRunResult:
        """Konu hakkında tam rapor oluştur - kalite kontrolü ile"""

        logger.info(f"Rapor oluşturma başlatılıyor: {topic}")

        self._reset_search_metadata()
        fallback_messages: List[str] = []

        attempted = list(attempted_llms or [])
        attempted.append(self.llm_provider_id)

        state = ReportAgentState(topic=topic)

        try:
            result = await self.graph.ainvoke(state)

            if isinstance(result, dict):
                final_report = result.get("final_report", "")
                quality_metadata = result.get("quality_metadata", {})
            else:
                final_report = getattr(result, "final_report", "")
                quality_metadata = getattr(result, "quality_metadata", {})

            search_fallbacks, used_search_ids = self._gather_search_feedback()
            fallback_messages = _merge_unique_messages(fallback_messages, search_fallbacks)

            if final_report:
                if quality_metadata:
                    score = quality_metadata.get("overall_score", "unknown")
                    logger.info(f"Rapor başarıyla oluşturuldu - Kalite skoru: {score}")
                else:
                    logger.info("Rapor başarıyla oluşturuldu")

                return ReportRunResult(
                    content=final_report,
                    fallback_messages=fallback_messages,
                    llm_provider_id=self.llm_provider_id,
                    search_provider_ids=used_search_ids,
                )

            logger.error("Rapor oluşturulamadı")
            return ReportRunResult(
                content="",
                error="Rapor oluşturulurken bir hata oluştu.",
                fallback_messages=fallback_messages,
                llm_provider_id=self.llm_provider_id,
                search_provider_ids=used_search_ids,
            )

        except ProviderRateLimitError as exc:
            return await self._handle_rate_limit_error(
                exc,
                topic,
                fallback_messages,
                attempted,
            )

        except Exception as e:
            logger.error(f"Rapor oluşturma hatası: {e}")
            search_fallbacks, used_search_ids = self._gather_search_feedback()
            fallback_messages = _merge_unique_messages(fallback_messages, search_fallbacks)
            return ReportRunResult(
                content="",
                error=f"Hata: {str(e)}",
                fallback_messages=fallback_messages,
                llm_provider_id=self.llm_provider_id,
                search_provider_ids=used_search_ids,
            )

    async def save_report(self, report_content: str, filename: str = None):
        """Raporu dosyaya kaydet"""
        from report_agent_setup import REPORT_OUTPUT_DIR

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rapor_{timestamp}.md"

        # Çıktı dizinini kontrol et
        filepath = os.path.join(REPORT_OUTPUT_DIR, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Rapor kaydedildi: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Rapor kaydetme hatası: {e}")
            return None


# CLI Interface
class ReportAgentCLI:
    """Komut satırı arayüzü"""

    def __init__(self):
        self.agent = MainReportAgent()

    async def interactive_mode(self):
        """İnteraktif mod"""
        print("🤖 Rapor Ajanına Hoş Geldiniz!")
        print("=" * 50)

        while True:
            try:
                print("\n📝 Rapor oluşturmak istediğiniz konuyu girin:")
                print("(Çıkmak için 'exit' yazın)")

                topic = input("> ").strip()

                if topic.lower() in ['exit', 'çıkış', 'quit']:
                    print("Güle güle! 👋")
                    break

                if not topic:
                    print("❌ Lütfen bir konu girin.")
                    continue

                print(f"\n🔍 '{topic}' konusunda rapor oluşturuluyor...")
                print("Bu işlem birkaç dakika sürebilir. Lütfen bekleyin...")

                result = await self.agent.generate_report(topic)

                if result.error:
                    print(f"❌ {result.error}")
                    if result.fallback_messages:
                        print("ℹ️ Sağlayıcı notları:")
                        for note in result.fallback_messages:
                            print(f" - {note}")
                    continue

                if result.fallback_messages:
                    print("ℹ️ Sağlayıcı notları:")
                    for note in result.fallback_messages:
                        print(f" - {note}")

                print("\n" + "="*70)
                print("📊 OLUŞTURULAN RAPOR")
                print("="*70)
                print(result.content)
                print("="*70)

                # Kaydetme seçeneği
                save_choice = input("\n💾 Raporu dosyaya kaydetmek ister misiniz? (e/h): ").lower()
                if save_choice in ['e', 'evet', 'y', 'yes']:
                    filename = await self.agent.save_report(result.content)
                    if filename:
                        print(f"✅ Rapor kaydedildi: {filename}")
                    else:
                        print("❌ Rapor kaydedilemedi.")

            except KeyboardInterrupt:
                print("\n\n👋 Program sonlandırılıyor...")
                break
            except Exception as e:
                print(f"\n❌ Hata oluştu: {e}")
                logger.error(f"CLI hatası: {e}")


# Ana çalıştırma fonksiyonu
async def main():
    """Ana fonksiyon"""
    print("🚀 Rapor Ajanı Sistemi")
    print("Geliştirici: AI Assistant")
    print("Kodlama: UTF-8")
    print("Platform: Windows Uyumlu")
    print("-" * 40)

    # .env dosyası ve API key kontrolü
    from report_agent_setup import check_api_keys

    if not check_api_keys():
        print("\n📋 Lütfen .env dosyasını oluşturun ve API key'lerinizi ekleyin:")
        print("OPENROUTER_API_KEY=your_openrouter_key")
        print("TAVILY_API_KEY=your_tavily_key")
        return

    print("✅ Sistem hazır!")

    # CLI başlat
    cli = ReportAgentCLI()
    await cli.interactive_mode()


if __name__ == "__main__":
    # Windows için event loop ayarı
    import sys
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
