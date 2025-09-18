# -*- coding: utf-8 -*-
"""
Yazar Ajan - Araştırma verilerini kullanarak rapor bölümleri yazar
"""

import asyncio
from typing import Dict, Any, Optional, TypedDict
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

# TypedDict kullanarak LangGraph uyumlu state
class SectionWriterState(TypedDict, total=False):
    section_name: str
    section_description: str
    section_index: int
    research_data: Optional[str]
    additional_research: Optional[str]
    content: str
    needs_research: bool

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
            section_name = state.section_name if hasattr(state, 'section_name') else state.get('section_name', 'Bilinmeyen')
            logger.info(f"Araştırma ihtiyacı kontrol ediliyor: {section_name}")
            
            needs_research = state.needs_research if hasattr(state, 'needs_research') else state.get('needs_research', False)
            additional_research = state.additional_research if hasattr(state, 'additional_research') else state.get('additional_research')
            
            if needs_research and not additional_research:
                return "research"
            else:
                return "write"
        
        async def do_additional_research(state):
            """Bölüm için ek araştırma yap"""
            section_name = state.section_name if hasattr(state, 'section_name') else state.get('section_name', 'Bilinmeyen')
            section_description = state.section_description if hasattr(state, 'section_description') else state.get('section_description', '')
            
            logger.info(f"Ek araştırma yapılıyor: {section_name}")
            
            # Araştırma sorgularını oluştur
            research_messages = self.research_prompt.format_messages(
                section_name=section_name,
                section_description=section_description
            )
            
            response = await self.llm_with_tools.ainvoke(research_messages)
            
            # Eğer araç çağrısı varsa çalıştır
            additional_research = ""
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call['name'] == 'search_web':
                        import json
                        args = json.loads(tool_call['args'])
                        search_result = await self.search_tool.ainvoke(args)
                        additional_research += "\n\n" + search_result
            
            # State güncelle
            if hasattr(state, '__dict__'):
                state.additional_research = (state.additional_research or "") + additional_research
                return state
            else:
                updated_state = dict(state)
                updated_state['additional_research'] = updated_state.get('additional_research', '') + additional_research
                return updated_state
        
        async def write_section(state):
            """Bölümü yaz"""
            section_name = state.section_name if hasattr(state, 'section_name') else state.get('section_name', 'Bilinmeyen')
            section_description = state.section_description if hasattr(state, 'section_description') else state.get('section_description', '')
            section_index = state.section_index if hasattr(state, 'section_index') else state.get('section_index', 1)
            research_data = state.research_data if hasattr(state, 'research_data') else state.get('research_data', '')
            additional_research = state.additional_research if hasattr(state, 'additional_research') else state.get('additional_research', '')
            
            logger.info(f"Bölüm yazılıyor: {section_name}")
            
            # Tüm araştırma verilerini birleştir
            all_research = ""
            if research_data:
                all_research += research_data
            if additional_research:
                all_research += "\n\n=== EK ARAŞTIRMA ===\n" + additional_research
            
            # Yazma promptunu hazırla
            writer_messages = self.writer_prompt.format_messages(
                section_name=section_name,
                section_description=section_description,
                section_index=section_index,
                research_data=all_research or "Mevcut araştırma verisi yok."
            )
            
            response = await self.llm.ainvoke(writer_messages)
            content = response.content
            
            logger.info(f"Bölüm tamamlandı: {section_name}")
            
            # State güncelle
            if hasattr(state, '__dict__'):
                state.content = content
                return state
            else:
                updated_state = dict(state)
                updated_state['content'] = content
                return updated_state
        
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
        
        state: SectionWriterState = {
            "section_name": section_name,
            "section_description": section_description,
            "section_index": section_index,
            "research_data": research_data,
            "additional_research": None,
            "content": "",
            "needs_research": needs_research
        }
        
        result = await self.graph.ainvoke(state)
        
        logger.info(f"Bölüm yazma tamamlandı: {section_name}")
        
        # Result'dan content'i al
        if isinstance(result, dict) and 'content' in result:
            return result['content']
        elif hasattr(result, 'content'):
            return result.content
        else:
            logger.error(f"Content bulunamadı. Result tipi: {type(result)}, Keys: {list(result.keys()) if isinstance(result, dict) else 'No keys'}")
            return f"# {section_name}\n\nBu bölüm oluşturulurken bir hata oluştu."

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
