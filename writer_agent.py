# -*- coding: utf-8 -*-
"""
Yazar Ajan - Araştırma verilerini kullanarak rapor bölümleri yazar
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, Any, Optional, TypedDict, List
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

                        raw_args = tool_call.get('args', {})

                        if isinstance(raw_args, dict):
                            args = raw_args
                        elif isinstance(raw_args, str):
                            raw_args = raw_args.strip()
                            if raw_args:
                                try:
                                    parsed_args = json.loads(raw_args)
                                except json.JSONDecodeError:
                                    logger.warning(
                                        "Araç argümanları JSON olarak parse edilemedi, string kullanılacak: %s",
                                        raw_args,
                                    )
                                    parsed_args = raw_args
                                if isinstance(parsed_args, dict):
                                    args = parsed_args
                                elif isinstance(parsed_args, list):
                                    args = {"queries": parsed_args}
                                else:
                                    args = {"queries": [parsed_args]}
                            else:
                                args = {}
                        else:
                            logger.warning(
                                "Araç argümanları beklenmeyen tipte: %s", type(raw_args).__name__
                            )
                            args = {}

                        search_result = await self.search_tool.ainvoke(args)
                        additional_research += "\n\n" + search_result
            
            # State güncelle
            if hasattr(state, '__dict__'):
                state.additional_research = (state.additional_research or "") + additional_research
                return state
            else:
                updated_state = dict(state)
                existing_research = updated_state.get('additional_research') or ""
                updated_state['additional_research'] = existing_research + additional_research
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

        if not sections:
            logger.warning("Derlenecek bölüm bulunamadı, manuel derleme kullanılacak")
            return self._manual_compile(topic, sections)

        # Bölümleri formatla
        formatted_sections = ""
        for i, section in enumerate(sections, 1):
            formatted_sections += f"=== BÖLÜM {i} ===\n{section}\n\n"

        messages = self.compile_prompt.format_messages(
            topic=topic,
            sections=formatted_sections
        )

        try:
            response = await self.llm.ainvoke(messages)
        except Exception as exc:
            logger.error(
                "LLM tabanlı derleme başarısız oldu, manuel derleme kullanılacak: %s",
                exc,
            )
            return self._manual_compile(topic, sections)

        stop_reason: Optional[str] = None
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            stop_reason = response_metadata.get("stop_reason") or response_metadata.get("finish_reason")

        if stop_reason is None:
            additional_kwargs = getattr(response, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict):
                stop_reason = additional_kwargs.get("stop_reason") or additional_kwargs.get("finish_reason")

        usage_info = getattr(response, "usage_metadata", None)
        if usage_info is None and isinstance(response_metadata, dict):
            usage_info = response_metadata.get("usage")

        compiled_report = response.content or ""

        fallback_reason = self._should_use_fallback(compiled_report, sections, stop_reason, usage_info)
        if fallback_reason:
            logger.warning(
                "LLM derleme çıktısı hedeflenen kapsamı karşılamadı (%s). Manuel derleme uygulanıyor.",
                fallback_reason,
            )
            return self._manual_compile(topic, sections)

        if stop_reason in {"max_tokens", "length"}:
            logger.warning(
                "Rapor derleme çıktısı token limitinde kesilmiş olabilir (stop_reason=%s, usage=%s).",
                stop_reason,
                usage_info,
            )

        logger.info("Rapor derleme tamamlandı")
        return compiled_report

    def _should_use_fallback(
        self,
        compiled_report: str,
        sections: list,
        stop_reason: Optional[str],
        usage_info: Optional[Any],
    ) -> Optional[str]:
        """LLM çıktısının yeterli olup olmadığını değerlendir."""

        if stop_reason in {"max_tokens", "length"}:
            return f"stop_reason={stop_reason}"

        total_section_words = sum(len(str(section).split()) for section in sections)
        compiled_words = len(compiled_report.split())

        if total_section_words >= 400:
            expected_min = max(400, int(total_section_words * 0.5))
            if compiled_words < expected_min:
                return f"compiled_report_too_short({compiled_words} < {expected_min})"

        # İçindekiler/Giriş/Sonuç başlıklarının eksikliği rapor bütünlüğünü bozabilir
        normalized_report = compiled_report.lower()
        critical_markers = ["içindekiler", "giriş", "sonuç"]
        missing_markers = [marker for marker in critical_markers if marker not in normalized_report]
        if len(missing_markers) >= 2 and total_section_words >= 200:
            return f"missing_markers={','.join(missing_markers)}"

        return None

    def _manual_compile(self, topic: str, sections: list) -> str:
        """LLM başarısız olduğunda manuel olarak rapor derle."""

        topic_text = topic or "Genel Konu"
        section_infos = []

        for index, raw_section in enumerate(sections, 1):
            section_text = str(raw_section) if not isinstance(raw_section, str) else raw_section
            title, body = self._extract_section_title_and_body(section_text, index)
            section_infos.append((title, body))

        if not section_infos:
            section_infos.append(("Araştırma Özeti", "Bu rapor için bölüm içerikleri sağlanamadı."))

        section_titles = [title for title, _ in section_infos]
        today = datetime.now().strftime("%d %B %Y")

        lines: List[str] = [
            f"# {topic_text} - Kapsamlı Araştırma Raporu",
            f"_Yayın Tarihi: {today}_",
            "",
            "## İçindekiler",
            "",
        ]

        for index, (title, _) in enumerate(section_infos, 1):
            anchor = self._slugify_title(title, index)
            lines.append(f"{index}. [{title}](#{anchor})")

        lines.append("")

        intro_sections_text = ""
        if section_titles:
            if len(section_titles) == 1:
                intro_sections_text = section_titles[0]
            else:
                intro_sections_text = ", ".join(section_titles[:-1]) + f" ve {section_titles[-1]}"

        lines.append("## Giriş")
        lines.append("")

        intro_paragraphs = [
            f"Bu rapor, {topic_text} başlığını derinlemesine incelemek amacıyla hazırlanmıştır.",
        ]

        if intro_sections_text:
            intro_paragraphs.append(
                f"Çalışma kapsamında {intro_sections_text} başlıkları ayrıntılı olarak değerlendirilmiştir."
            )

        intro_paragraphs.append(
            "Her bölüm güncel araştırma bulguları, uygulama içgörüleri ve stratejik değerlendirmelerle desteklenmiştir."
        )

        for paragraph in intro_paragraphs:
            lines.append(paragraph)
            lines.append("")

        for index, (title, body) in enumerate(section_infos, 1):
            lines.append(f"## {index}. {title}")
            lines.append("")
            cleaned_body = body.strip()
            if cleaned_body:
                lines.append(cleaned_body)
            else:
                lines.append("_Bu bölüm için detaylı içerik sağlanamadı._")
            lines.append("")

        lines.append("## Sonuç ve Öneriler")
        lines.append("")

        if section_titles:
            section_summary = ", ".join(section_titles)
        else:
            section_summary = topic_text

        lines.append(
            f"Rapor genelinde ele alınan {len(section_infos)} bölüm, {section_summary} başlıkları üzerinden bütünsel bir çerçeve sunmaktadır."
        )
        lines.append(
            "Bu bulgular, kurumların stratejik planlama, teknoloji yatırımları ve operasyonel iyileştirme süreçleri için yol gösterici niteliktedir."
        )
        lines.append("")
        lines.append("### Önerilen Adımlar")
        lines.append("")
        lines.append("- Bulguları periyodik olarak güncelleyecek bir izleme ve değerlendirme mekanizması kurun.")
        lines.append("- Paydaşlarla paylaşılacak atölye ve bilgilendirme oturumları planlayın.")
        lines.append("- Öncelikli fırsat ve risklere yönelik pilot projeler tasarlayarak öğrenimleri hızlandırın.")

        while lines and lines[-1] == "":
            lines.pop()

        return "\n".join(lines)

    @staticmethod
    def _extract_section_title_and_body(section_text: str, index: int) -> tuple[str, str]:
        """Bölüm içeriğinden başlık ve gövdeyi ayıkla."""

        if not section_text.strip():
            return f"Bölüm {index}", ""

        lines = section_text.strip().splitlines()
        heading_pattern = re.compile(r"^\s*(#{1,6})\s+(.*)$")

        for line_index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            match = heading_pattern.match(line)
            if match:
                title = match.group(2).strip() or f"Bölüm {index}"
                body = "\n".join(lines[line_index + 1 :]).strip()
                return title, body

            title = stripped.lstrip("#").strip() or f"Bölüm {index}"
            body = "\n".join(lines[line_index + 1 :]).strip()
            return title, body

        return f"Bölüm {index}", section_text.strip()

    @staticmethod
    def _slugify_title(title: str, index: int) -> str:
        """Başlıktan markdown anchor üret."""

        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        if not slug:
            slug = f"bolum-{index}"
        return slug

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
