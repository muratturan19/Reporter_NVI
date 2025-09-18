# -*- coding: utf-8 -*-
"""
Ana Rapor Ajanı - Tüm bileşenleri birleştiren tam sistem
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import logging

# Diğer modüllerden import
from report_agent_setup import create_llm, search_web, Section, ReportStructure
from researcher_agent import ResearcherAgent
from writer_agent import WriterAgent, ReportCompiler

logger = logging.getLogger(__name__)


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
    return data


def _get_first_value(data: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    """Verilen anahtar listesinden ilk mevcut değeri döndür."""

    for key in keys:
        if key in data and data[key]:
            return data[key]
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
REPORT_PLANNER_PROMPT = """
Sen deneyimli bir rapor planlama uzmanısın. Verilen konu ve araştırma verilerini kullanarak detaylı rapor yapısı oluşturuyorsun.

ÖNEMLİ: Yanıtını SADECE JSON formatında ver, başka hiçbir açıklama ekleme.

Rapor planlarken:
1. Konuyu kapsamlı şekilde ele al
2. 4-6 bölümden oluşan mantıklı bir akış oluştur
3. Her bölüm için net açıklama yaz
4. Türk okuyucu kitlesini göz önünde bulundur

SADECE ŞU FORMATTA YANIT VER:
{
  "title": "Rapor Başlığı",
  "sections": [
    {
      "name": "Bölüm Adı",
      "description": "Bu bölümde neyin ele alınacağının detaylı açıklaması",
      "research": true
    },
    {
      "name": "İkinci Bölüm",
      "description": "İkinci bölümün açıklaması",
      "research": false
    }
  ]
}
"""

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


class MainReportAgent:
    """Ana Rapor Ajan Sınıfı - Tüm sistemi yönetir"""

    def __init__(self):
        # Model ve araçları başlat
        self.llm = create_llm()
        self.search_tool = search_web

        # Alt ajanları başlat
        self.researcher = ResearcherAgent(self.llm, self.search_tool)
        self.writer = WriterAgent(self.llm, self.search_tool)
        self.compiler = ReportCompiler(self.llm)

        # Planlama promptu
        self.planner_prompt = ChatPromptTemplate.from_messages([
            ("system", REPORT_PLANNER_PROMPT),
            ("human", PLANNER_HUMAN_PROMPT)
        ])

        # Ana grafı oluştur
        self.graph = self._build_main_graph()

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

            messages = self.planner_prompt.format_messages(
                topic=state.topic,
                research_data=state.research_data[:2000]  # İlk 2000 karakter
            )

            response = await self.llm.ainvoke(messages)

            try:
                # JSON yanıtını parse et - daha güvenli yöntem
                content = response.content.strip()

                # JSON bloğunu bul
                json_start = content.find('{')
                json_end = content.rfind('}') + 1

                if json_start == -1 or json_end <= json_start:
                    raise ValueError("Geçerli JSON bulunamadı")

                json_str = content[json_start:json_end]
                plan_data_raw = json.loads(json_str)
                plan_data = _normalize_plan_response(plan_data_raw)

                title = _get_first_value(
                    plan_data,
                    ["title", "rapor başlığı", "rapor_baslığı", "rapor basligi", "başlık"],
                )
                raw_sections = plan_data.get("sections") or plan_data.get("bölümler")

                if not title or not isinstance(raw_sections, list):
                    raise ValueError("Plan çıktısı beklenen alanları içermiyor")

                sections: List[Section] = []
                for raw_section in raw_sections:
                    section = _parse_section_entry(raw_section)
                    if section:
                        sections.append(section)

                if not sections:
                    raise ValueError("Plan çıktısında kullanılabilir bölüm bulunamadı")

                state.report_structure = ReportStructure(title=title, sections=sections)
                logger.info(f"Rapor planlandı: {len(sections)} bölüm")

            except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                logger.error(f"Rapor planlama hatası: {e}")
                # Varsayılan yapı oluştur
                state.report_structure = ReportStructure(
                    title=f"{state.topic or 'Genel'} Raporu",
                    sections=[
                        Section("Giriş", "Konuya genel bakış", False),
                        Section("Detaylar", "Konunun detaylı analizi", True),
                        Section("Sonuç", "Bulgular ve öneriler", False)
                    ]
                )

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

        # Graf oluştur
        workflow = StateGraph(ReportAgentState)

        # Düğümler
        workflow.add_node("research", initial_research)
        workflow.add_node("plan", plan_report)
        workflow.add_node("write", write_sections)
        workflow.add_node("compile", compile_final_report)

        # Kenarlar
        workflow.add_edge(START, "research")
        workflow.add_edge("research", "plan")
        workflow.add_edge("plan", "write")
        workflow.add_edge("write", "compile")
        workflow.add_edge("compile", END)

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

    async def generate_report(self, topic: str) -> str:
        """Konu hakkında tam rapor oluştur"""
        logger.info(f"Rapor oluşturma başlatılıyor: {topic}")

        state = ReportAgentState(topic=topic)

        try:
            result = await self.graph.ainvoke(state)

            if result.final_report:
                logger.info("Rapor başarıyla oluşturuldu")
                return result.final_report
            else:
                logger.error("Rapor oluşturulamadı")
                return "Rapor oluşturulurken bir hata oluştu."

        except Exception as e:
            logger.error(f"Rapor oluşturma hatası: {e}")
            return f"Hata: {str(e)}"

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

                report = await self.agent.generate_report(topic)

                print("\n" + "="*70)
                print("📊 OLUŞTURULAN RAPOR")
                print("="*70)
                print(report)
                print("="*70)

                # Kaydetme seçeneği
                save_choice = input("\n💾 Raporu dosyaya kaydetmek ister misiniz? (e/h): ").lower()
                if save_choice in ['e', 'evet', 'y', 'yes']:
                    filename = await self.agent.save_report(report)
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
