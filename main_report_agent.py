# -*- coding: utf-8 -*-
"""
Ana Rapor Ajanı - Tüm bileşenleri birleştiren tam sistem
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import logging

# Diğer modüllerden import
from report_agent_setup import create_llm, search_web, Section, ReportStructure
from researcher_agent import ResearcherAgent
from writer_agent import WriterAgent, ReportCompiler

logger = logging.getLogger(__name__)

# Planlama promptı
REPORT_PLANNER_PROMPT = """
Sen deneyimli bir rapor planlama uzmanısın. Verilen konu ve araştırma verilerini kullanarak detaylı rapor yapısı oluşturuyorsun.

Rapor planlarken:
1. Konuyu kapsamlı şekilde ele al
2. Mantıklı bir akış oluştur
3. Her bölüm için net açıklama yaz
4. Bölümlerin birbiriyle uyumunu sağla
5. Türk okuyucu kitlesini göz önünde bulundur

Bölümler şu formatta olmalı:
```json
{
  "title": "Rapor Başlığı",
  "sections": [
    {
      "name": "Bölüm Adı",
      "description": "Bu bölümde neyin ele alınacağının detaylı açıklaması",
      "research": true/false
    }
  ]
}
```
"""

PLANNER_HUMAN_PROMPT = """
Konu: {topic}

Araştırma Verileri:
{research_data}

Bu bilgileri kullanarak 4-6 bölümden oluşan kapsamlı bir rapor yapısı oluştur. JSON formatında yanıtla.
"""

class ReportAgentState:
    """Ana rapor ajan durumu"""
    def __init__(self):
        self.topic: str = ""
        self.research_data: str = ""
        self.report_structure: Optional[ReportStructure] = None
        self.sections_content: List[str] = []
        self.final_report: str = ""
        self.messages: List[Any] = []

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
                # JSON yanıtını parse et
                json_start = response.content.find('{')
                json_end = response.content.rfind('}') + 1
                json_str = response.content[json_start:json_end]
                
                plan_data = json.loads(json_str)
                
                sections = []
                for section_data in plan_data["sections"]:
                    sections.append(Section(
                        name=section_data["name"],
                        description=section_data["description"],
                        research=section_data.get("research", False)
                    ))
                
                state.report_structure = ReportStructure(
                    title=plan_data["title"],
                    sections=sections
                )
                
                logger.info(f"Rapor planlandı: {len(sections)} bölüm")
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Rapor planlama hatası: {e}")
                # Varsayılan yapı oluştur
                state.report_structure = ReportStructure(
                    title=f"{state.topic} Raporu",
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
            
            sections_content = []
            
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
    
    async def generate_report(self, topic: str) -> str:
        """Konu hakkında tam rapor oluştur"""
        logger.info(f"Rapor oluşturma başlatılıyor: {topic}")
        
        state = ReportAgentState()
        state.topic = topic
        
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