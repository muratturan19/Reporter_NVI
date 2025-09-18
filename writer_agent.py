# -*- coding: utf-8 -*-
"""
Yazar Ajan - Araştırma verilerini kullanarak rapor bölümleri yazar
"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import logging

logger = logging.getLogger(__name__)

# Yazma promptları
SECTION_WRITER_SYSTEM_PROMPT = """
Sen profesyonel bir teknik rapor yazarısın. Verilen araştırma verilerini kullanarak yüksek kaliteli, bilgilendirici rapor bölümleri yazıyorsun.

Yazarken dikkat etmen gerekenler:
1. Türkçe dilbilgisi kurallarına uygun yazma
2. Teknik terimleri doğru kullanma
3. Araştırma verilerini referans gösterme
4. Akıcı ve anlaşılır dil kullanma
5. Markdown formatında yazma
6. Başlık hiyerarşisini koruma
7. Gerektiğinde örnek ve açıklamalar ekleme

Yazım kuralları:
- H2 başlıkları (##) bölüm başlıkları için
- H3 başlıkları (###) alt bölümler için
- Kalın yazı (**metin**) önemli kavramlar için
- İtalik (*metin*) vurgulamalar için
- Madde işaretleri (-) listeler için
- Numaralı listeler (1.) adımlar için
"""

SECTION_WRITER_HUMAN_PROMPT = """
Bölüm Bilgileri:
- İsim: {section_name}
- Açıklama: {section_description}
- Sıra: {section_index}

Araştırma Verileri:
{research_data}

Bu bilgileri kullanarak profesyonel bir rapor bölümü yaz. Bölüm yaklaşık 300-500 kelime olmalı ve Markdown formatında olmalı.
"""

RESEARCH_PROMPT = """
"{section_name}" bölümü için ek araştırma gerekiyor.

Bölüm açıklaması: {section_description}

Bu bölüm için 2-3 spesifik web arama sorgusu oluştur. Sorgular bu bölümün içeriğini destekleyecek detaylı bilgileri hedeflemeli.
"""

@dataclass
class SectionWriterState:
    """Bölüm yazar durumu"""
    section_name: str
    section_description: str
    section_index: int
    research_data: Optional[str] = None
    additional_research: Optional[str] = None
    content: str = ""
    needs_research: bool = False

class WriterAgent:
    """Yazar Ajan Sınıfı"""
    
    def __init__(self, llm, search_tool):
        self.llm = llm
        self.search_tool = search_tool
        self.llm_with_tools = llm.bind_tools([search_tool])
        
        # Prompt template'leri
        self.writer_prompt = ChatPromptTemplate.from_messages([
            ("system", SECTION_WRITER_SYSTEM_PROMPT),
            ("human", SECTION_WRITER_HUMAN_PROMPT)
        ])
        
        self.research_prompt = ChatPromptTemplate.from_messages([
            ("system", "Sen araştırma sorguları oluşturan bir uzmansın."),
            ("human", RESEARCH_PROMPT)
        ])
        
        # Graf oluştur
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Yazar ajan grafını oluştur"""
        
        async def check_research_need(state: SectionWriterState):
            """Ek araştırma gerekip gerekmediğini kontrol et"""
            logger.info(f"Araştırma ihtiyacı kontrol ediliyor: {state.section_name}")
            
            if state.needs_research and not state.additional_research:
                return "research"
            else:
                return "write"
        
        async def do_additional_research(state: SectionWriterState):
            """Bölüm için ek araştırma yap"""
            logger.info(f"Ek araştırma yapılıyor: {state.section_name}")
            
            # Araştırma sorgularını oluştur
            research_messages = self.research_prompt.format_messages(
                section_name=state.section_name,
                section_description=state.section_description
            )
            
            response = await self.llm_with_tools.ainvoke(research_messages)
            
            # Eğer araç çağrısı varsa çalıştır
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call['name'] == 'search_web':
                        import json
                        args = json.loads(tool_call['args'])
                        search_result = await self.search_tool.ainvoke(args)
                        
                        # Ek araştırma verilerini birleştir
                        existing_research = state.additional_research or ""
                        state.additional_research = existing_research + "\n\n" + search_result
            
            return state
        
        async def write_section(state: SectionWriterState):
            """Bölümü yaz"""
            logger.info(f"Bölüm yazılıyor: {state.section_name}")
            
            # Tüm araştırma verilerini birleştir
            all_research = ""
            if state.research_data:
                all_research += state.research_data
            if state.additional_research:
                all_research += "\n\n=== EK ARAŞTIRMA ===\n" + state.additional_research
            
            # Yazma promptunu hazırla
            writer_messages = self.writer_prompt.format_messages(
                section_name=state.section_name,
                section_description=state.section_description,
                section_index=state.section_index,
                research_data=all_research or "Mevcut araştırma verisi yok."
            )
            
            response = await self.llm.ainvoke(writer_messages)
            state.content = response.content
            
            logger.info(f"Bölüm tamamlandı: {state.section_name}")
            return state
        
        # Graf oluştur
        workflow = StateGraph(SectionWriterState)
        
        # Düğümler
        workflow.add_node("research", do_additional_research)
        workflow.add_node("write", write_section)
        
        # Başlangıç koşullu
        workflow.add_conditional_edges(
            START,
            check_research_need,
            {
                "research": "research",
                "write": "write"
            }
        )
        
        # Araştırmadan yazmaya
        workflow.add_edge("research", "write")
        
        # Yazmadan bitir
        workflow.add_edge("write", END)
        
        return workflow.compile()
    
    async def write_section(
        self,
        section_name: str,
        section_description: str,
        section_index: int,
        research_data: Optional[str] = None,
        needs_research: bool = False
    ) -> str:
        """Bölüm yaz"""
        logger.info(f"Bölüm yazma başlatılıyor: {section_name}")
        
        state = SectionWriterState(
            section_name=section_name,
            section_description=section_description,
            section_index=section_index,
            research_data=research_data,
            needs_research=needs_research
        )
        
        result = await self.graph.ainvoke(state)
        
        logger.info(f"Bölüm yazma tamamlandı: {section_name}")
        return result.content

# Ana rapor derleyici
class ReportCompiler:
    """Rapor derleyici sınıfı"""
    
    def __init__(self, llm):
        self.llm = llm
        
        self.compile_prompt = ChatPromptTemplate.from_messages([
            ("system", """
Sen profesyonel rapor derleyicisisin. Verilen bölümleri bir araya getirerek tutarlı, akıcı bir rapor oluşturuyorsun.

Derleme kuralları:
1. Rapor başlığı ve tarih ekle
2. İçindekiler tablosu oluştur
3. Bölümler arası geçişleri düzelt
4. Tutarlı stil ve format sağla
5. Sonuç ve öneri bölümü ekle
6. Markdown formatını koru
7. Türkçe dilbilgisi kurallarını uygula
            """),
            ("human", """
Rapor Konusu: {topic}

Bölümler:
{sections}

Bu bölümleri kullanarak profesyonel bir rapor derle. Rapor şu yapıda olmalı:
- Başlık ve tarih
- İçindekiler
- Giriş (otomatik oluştur)
- Verilen bölümler
- Sonuç ve öneriler (otomatik oluştur)
            """)
        ])
    
    async def compile_report(self, topic: str, sections: list) -> str:
        """Bölümleri birleştirerek final rapor oluştur"""
        logger.info("Rapor derleniyor...")
        
        # Bölümleri formatla
        formatted_sections = ""
        for i, section in enumerate(sections, 1):
            formatted_sections += f"=== BÖLÜM {i} ===\n{section}\n\n"
        
        messages = self.compile_prompt.format_messages(
            topic=topic,
            sections=formatted_sections
        )
        
        response = await self.llm.ainvoke(messages)
        
        logger.info("Rapor derleme tamamlandı")
        return response.content

# Test fonksiyonu
async def test_writer():
    """Yazar ajanını test et"""
    from report_agent_setup import create_llm, search_web
    
    # Model ve araçları yükle
    llm = create_llm()
    
    # Yazar oluştur
    writer = WriterAgent(llm, search_web)
    
    # Test bölümü yaz
    content = await writer.write_section(
        section_name="Yapay Zeka Ajanlarına Giriş",
        section_description="Yapay zeka ajanlarının tanımı, özellikleri ve temel çalışma prensipleri",
        section_index=1,
        research_data="AI ajanları otonom sistemlerdir ve karar verme yeteneklerine sahiptir.",
        needs_research=True
    )
    
    print("\n=== YAZILAN BÖLÜM ===")
    print(content)

if __name__ == "__main__":
    asyncio.run(test_writer())