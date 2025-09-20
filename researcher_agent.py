# -*- coding: utf-8 -*-
"""Katmanlı araştırma ajanı

Bu modül, araştırma sürecini çok katmanlı stratejiler, dinamik
follow-up sorguları ve kaynak kalitesi değerlendirmeleriyle
zenginleştirir. Ajan; temel, teknik, pratik, gelecek odaklı ve
karşılaştırmalı katmanlarda çalışarak sonuçları analiz eder ve sentezler.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from json_parser_fix import parse_json_from_response

logger = logging.getLogger(__name__)


RESEARCH_LAYER_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "foundation",
        "title": "Foundation · Temel Katman",
        "description": "Temel kavramlar, terminoloji, tarihçe ve pazar görünümü",
        "objectives": [
            "Temel kavramları ve terminolojiyi açıklamak",
            "Kavramsal çerçeveyi ve tarihsel gelişimi belgelemek",
            "Mevcut pazar durumunu ve ekosistemi özetlemek",
        ],
        "source_targets": [
            "academic papers",
            "government & standards bodies",
            "industry landscape reports",
        ],
        "seed_queries": [
            '"{topic}" fundamentals 2024',
            '"{topic}" history timeline',
            '"{topic}" market overview report 2024',
            '"{topic}" temel kavramlar nelerdir',
        ],
        "aliases": ["foundation", "temel", "fundamental"],
    },
    {
        "id": "technical",
        "title": "Technical · Mimari & Uygulama",
        "description": "Teknik mimari, altyapı ve uygulama ayrıntıları",
        "objectives": [
            "Ana teknolojileri, modelleri ve algoritmaları incelemek",
            "Sistem mimarilerini ve entegrasyon modellerini çıkarmak",
            "Referans implementasyonları ve kod örneklerini toplamak",
        ],
        "source_targets": [
            "technical documentation",
            "architecture whitepapers",
            "GitHub repositories",
        ],
        "seed_queries": [
            '"{topic}" reference architecture 2024',
            '"{topic}" implementation guide filetype:pdf',
            '"{topic}" API documentation site:docs',
            '"{topic}" github repository',
        ],
        "aliases": ["technical", "teknik", "engineering", "implementation"],
    },
    {
        "id": "practical",
        "title": "Practical · Vaka & İş Değeri",
        "description": "Gerçek projeler, iş etkisi ve yatırım geri dönüşü",
        "objectives": [
            "Gerçek dünya uygulamalarını ve vaka çalışmalarını derlemek",
            "ROI, maliyet ve iş değeri analizlerini toplamak",
            "Regülasyon, etik ve operasyonel hususları belirlemek",
        ],
        "source_targets": [
            "industry reports",
            "case studies",
            "news & government initiatives",
        ],
        "seed_queries": [
            '"{topic}" case study healthcare',
            '"{topic}" deployment ROI 2024',
            '"{topic}" pilot project government',
            '"{topic}" gerçek dünya uygulamaları',
        ],
        "aliases": ["practical", "business", "case", "uygulama"],
    },
    {
        "id": "future",
        "title": "Future · Trendler & Yol Haritası",
        "description": "Gelecek projeksiyonları, trendler ve riskler",
        "objectives": [
            "Emerging trendleri ve teknoloji yol haritalarını tespit etmek",
            "Riskleri, engelleri ve başarı faktörlerini analiz etmek",
            "Uzun vadeli fırsatları ve dönüşümleri belirlemek",
        ],
        "source_targets": [
            "conference proceedings",
            "futurist think-tank reports",
            "patent databases",
        ],
        "seed_queries": [
            '"{topic}" future roadmap 2025',
            '"{topic}" emerging trends 2024',
            '"{topic}" risk assessment report',
            '"{topic}" patent landscape',
        ],
        "aliases": ["future", "trend", "roadmap", "gelecek"],
    },
    {
        "id": "comparative",
        "title": "Comparative · Karşılaştırmalı Analiz",
        "description": "Alternatif yaklaşımlar, benchmark ve karşılaştırmalar",
        "objectives": [
            "Rakip veya alternatif çözümleri karşılaştırmak",
            "Benchmark sonuçlarını ve metodolojilerini toplamak",
            "Avantaj/dezavantaj ve seçim kriterlerini belirlemek",
        ],
        "source_targets": [
            "comparative whitepapers",
            "benchmark studies",
            "independent analyst reviews",
        ],
        "seed_queries": [
            '"{topic}" vs alternative solutions 2024',
            '"{topic}" benchmarking study',
            '"{topic}" karşılaştırma raporu',
            '"{topic}" alternative approaches analysis',
        ],
        "aliases": ["comparative", "benchmark", "comparison", "alternatif"],
    },
]


SOURCE_SIGNAL_RULES: List[Dict[str, Any]] = [
    {
        "type": "academic",
        "score": 3,
        "signal": "Hakemli/Akademik kaynak",
        "patterns": [
            "arxiv.org",
            ".edu",
            ".ac.",
            "ieee.org",
            "acm.org",
            "springer",
            "nature.com",
            "sciencedirect",
            "frontiersin",
            "ieeexplore.ieee.org",
            "scholar.google",
            "researchgate",
            "pubmed",
        ],
    },
    {
        "type": "conference",
        "score": 3,
        "signal": "Konferans bildirisi",
        "patterns": [
            "proceedings",
            "conference",
            "neurips",
            "icml",
            "cvpr",
            "aaai",
            "emnlp",
        ],
    },
    {
        "type": "industry",
        "score": 2,
        "signal": "Endüstri raporu",
        "patterns": [
            "gartner",
            "mckinsey",
            "bcg",
            "accenture",
            "deloitte",
            "forrester",
            "pwc",
        ],
    },
    {
        "type": "technical",
        "score": 2,
        "signal": "Teknik dokümantasyon",
        "patterns": [
            "docs.",
            "documentation",
            "developer.",
            "readthedocs",
            "learn.microsoft",
            "docs.oracle.com",
            "aws.amazon.com",
            "cloud.google",
            "azure.microsoft",
        ],
    },
    {
        "type": "repository",
        "score": 2,
        "signal": "Kod deposu",
        "patterns": ["github.com", "gitlab.com", "bitbucket.org"],
    },
    {
        "type": "patent",
        "score": 3,
        "signal": "Patent verisi",
        "patterns": ["patents.google.com", "uspto.gov", "epo.org", "wipo.int"],
    },
    {
        "type": "government",
        "score": 2,
        "signal": "Resmi düzenleyici kaynak",
        "patterns": [".gov", "europa.eu", "who.int", "nih.gov", "ema.europa.eu"],
    },
    {
        "type": "news",
        "score": 1,
        "signal": "Güncel haber/analiz",
        "patterns": ["news", "forbes", "bloomberg", "reuters", "techcrunch", "wired"],
    },
]


TYPE_PRIORITY = {
    "academic": 6,
    "conference": 5,
    "patent": 5,
    "government": 4,
    "industry": 4,
    "technical": 3,
    "repository": 3,
    "news": 1,
}


LAYERED_QUERY_SYSTEM_PROMPT = """
Sen kıdemli bir araştırma stratejistisin. Verilen konu için çok
katmanlı bir araştırma planı hazırlayacaksın.

Her katman için:
- Katmanın odaklandığı açıyı 1-2 cümleyle özetle.
- Öncelikli kaynak tiplerini (akademik makale, patent, dokümantasyon vb.) belirt.
- Gelişmiş arama operatörlerini (site:, filetype:, intitle:, "\"exact\"")
  kullanan, güncel (>=2023) ve güvenilir kaynakları hedefleyen sorgular üret.
- Türkçe ve İngilizce anahtar kelimeleri birlikte değerlendir.

Yanıtını şu JSON formatında ver:
{{
  "layers": {{
    "foundation": {{
      "focus": "...",
      "source_targets": ["..."],
      "queries": ["..."]
    }},
    ...
  }},
  "notes": "Opsiyonel genel notlar",
  "total_queries": 0
}}

Belirtilen katman kimliklerini (foundation, technical, practical,
future, comparative) kullan. Her katman için minimum hedef sorgu
sayısını aşmamaya ama boş bırakmamaya dikkat et.
"""


LAYERED_QUERY_HUMAN_PROMPT = """
Konu: {topic}

Hedeflenen sorgu dağılımı:
{query_allocation}

Katman açıklamaları ve öncelikler:
{layer_instructions}

Lütfen yukarıdaki kriterlere göre JSON formatında çok katmanlı sorgu
planı üret.
"""


ANALYSIS_SYSTEM_PROMPT = """
Katmanlı arama çıktılarından sentez çıkaran uzman bir araştırma
analistisiniz. Aşağıdaki görevleri tamamlayın:
1. Katmanlar arasında tekrar eden örüntüleri, temaları ve içgörüleri belirleyin.
2. Çelişen iddiaları, belirsizlikleri ve doğrulama ihtiyaçlarını saptayın.
3. Her katmanda eksik kalan konuları tespit edin ve uzman kaynaklara yönelik
   1-3 yeni takip sorgusu önerin.
4. Ortaya çıkan trendleri, riskleri ve fırsatları özetleyin.
5. Kritik bulgular için hangi kaynaklarla çapraz doğrulama yapılması gerektiğini belirtin.

Yanıtınızı şu JSON formatında verin:
{{
  "patterns": [{{"insight": "...", "supporting_layers": ["..."]}}],
  "contradictions": [{{"issue": "...", "sources_to_compare": ["..."]}}],
  "trends": [{{"trend": "...", "evidence": ["..."]}}],
  "gaps": {{
    "foundation": {{
      "missing_topics": ["..."],
      "follow_up_queries": ["..."],
      "validation_targets": ["..."]
    }},
    ...
  }},
  "cross_validation": [{{"claim": "...", "recommended_sources": ["..."]}}]
}}

Eğer belirli bir bölüm için veri yoksa ilgili alanı boş liste olarak
bırakın. Tahmin yerine veriye dayalı öneriler üretin.
"""


ANALYSIS_HUMAN_PROMPT = """
Konu: {topic}

Sorgu planı özeti:
{query_plan_summary}

Öne çıkan kaynak adayları:
{top_sources}

İlk arama çıktısı özeti:
{search_digest}

Yukarıdaki bilgilere dayanarak talep edilen JSON formatını üretin.
"""


SYNTHESIS_SYSTEM_PROMPT = """
Katmanlı araştırma çıktıları, takip sorguları ve kalite değerlendirmesi
üzerinden bütüncül bir analiz raporu oluşturan uzman bir analistsiniz.
Sonuçları şu çerçevede sentezleyin:
- Katmanlar arası genel durum özeti
- Önemli örüntüler ve çıkarımlar
- Tespit edilen çelişkiler ve gerekli doğrulamalar
- Boşluklar ve önerilen ileri araştırma adımları
- Trendler, geleceğe yönelik beklentiler ve riskler
- En güvenilir kaynaklar ve neyi destekledikleri

Bulgu ve önerilerinizi veri dayanaklarıyla ilişkilendirin, kaynak tiplerini
belirtin ve karar vericilere yönelik aksiyon alınabilir tavsiyeler ekleyin.
"""


SYNTHESIS_HUMAN_PROMPT = """
Konu: {topic}

Sorgu planı özeti:
{query_plan_summary}

Analiz bulguları (JSON):
{analysis_json}

Takip araması özeti:
{follow_up_digest}

Birleştirilmiş arama çıktısı özeti:
{combined_digest}

En yüksek puanlı kaynaklar:
{top_sources}

Yukarıdaki girdileri kullanarak kapsamlı sentez raporu oluşturun.
"""


URL_PATTERN = re.compile(r"URL:\\s*(\\S+)")


class ResearcherAgent:
    """Çok katmanlı araştırma ajanı"""

    def __init__(self, llm, search_tool) -> None:
        self.llm = llm
        self.search_tool = search_tool
        self.research_layers = OrderedDict(
            (layer["id"], layer) for layer in RESEARCH_LAYER_DEFINITIONS
        )

        self.DEFAULT_SEARCH_TOPIC = "general"
        self.VALID_SEARCH_TOPICS = frozenset({"general", "news", "finance"})

        self.layered_query_prompt = ChatPromptTemplate.from_messages([
            ("system", LAYERED_QUERY_SYSTEM_PROMPT),
            ("human", LAYERED_QUERY_HUMAN_PROMPT),
        ])
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_SYSTEM_PROMPT),
            ("human", ANALYSIS_HUMAN_PROMPT),
        ])
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", SYNTHESIS_HUMAN_PROMPT),
        ])

    def _normalize_search_topic(self, topic: Optional[str]) -> str:
        """Arama aracı için konu değerini normalize et."""

        if topic is None:
            return self.DEFAULT_SEARCH_TOPIC

        normalized = str(topic).strip().lower()
        if not normalized:
            return self.DEFAULT_SEARCH_TOPIC

        if normalized in self.VALID_SEARCH_TOPICS:
            return normalized

        return self.DEFAULT_SEARCH_TOPIC

    async def research(self, topic: str, number_of_queries: Optional[int] = None) -> Dict[str, Any]:
        """Katmanlı araştırma sürecini yürüt."""

        if number_of_queries is None:
            from report_agent_setup import DEFAULT_SEARCH_QUERIES

            number_of_queries = DEFAULT_SEARCH_QUERIES

        logger.info("Katmanlı araştırma başlıyor: %s", topic)

        query_distribution = self._distribute_query_counts(number_of_queries)
        query_plan, plan_prompt_text, plan_response_text = await self._generate_layered_queries(
            topic, query_distribution
        )

        messages: List[Any] = []
        if plan_prompt_text:
            messages.append(HumanMessage(content=plan_prompt_text))
        messages.append(AIMessage(content=plan_response_text))

        initial_results, initial_tool_messages = await self._execute_query_plan(
            topic, query_plan, variant="initial"
        )
        messages.extend(initial_tool_messages)

        initial_quality_scores = self._score_sources(initial_results)

        analysis_data, analysis_prompt_text, analysis_response_text = await self._analyze_initial_findings(
            topic, query_plan, initial_results, initial_quality_scores
        )
        if analysis_prompt_text:
            messages.append(HumanMessage(content=analysis_prompt_text))
        messages.append(AIMessage(content=analysis_response_text))

        follow_up_queries = self._collect_follow_up_queries(topic, analysis_data)
        follow_up_results: Dict[str, List[Dict[str, Any]]] = OrderedDict()
        follow_up_tool_messages: List[ToolMessage] = []
        if follow_up_queries:
            logger.info("Takip sorguları oluşturuldu: %s", follow_up_queries)
            follow_up_plan = OrderedDict(
                (
                    layer_id,
                    {
                        "title": self.research_layers[layer_id]["title"],
                        "focus": "Follow-up araştırma",
                        "queries": queries,
                    },
                )
                for layer_id, queries in follow_up_queries.items()
            )
            follow_up_results, follow_up_tool_messages = await self._execute_query_plan(
                topic, follow_up_plan, variant="follow_up"
            )
            messages.extend(follow_up_tool_messages)
        else:
            logger.info("Takip sorgusu gerekmedi veya oluşturulamadı.")

        combined_results = self._merge_layer_results(initial_results, follow_up_results)
        combined_quality_scores = self._score_sources(combined_results)

        follow_up_digest = self._build_search_digest(follow_up_results, max_chars=2500)
        combined_digest = self._build_search_digest(combined_results, max_chars=4500)
        top_sources_summary = self._format_source_summary(combined_quality_scores, limit=12)

        final_summary_text, synthesis_prompt_text = await self._synthesize(
            topic,
            query_plan,
            analysis_data,
            follow_up_digest,
            combined_digest,
            top_sources_summary,
        )

        if synthesis_prompt_text:
            messages.append(HumanMessage(content=synthesis_prompt_text))
        messages.append(AIMessage(content=final_summary_text))

        result = {
            "messages": messages,
            "query_plan": query_plan,
            "initial_results": initial_results,
            "follow_up_queries": follow_up_queries,
            "follow_up_results": follow_up_results,
            "analysis": analysis_data,
            "quality_scores": combined_quality_scores,
            "final_summary": final_summary_text,
        }

        logger.info("Araştırma süreci tamamlandı")
        return result

    async def _generate_layered_queries(
        self, topic: str, query_distribution: Dict[str, int]
    ) -> Tuple[OrderedDict, str, str]:
        """LLM üzerinden katmanlı sorgu planı üret."""

        allocation_lines = "\n".join(
            f"- {self.research_layers[layer_id]['title']} ({layer_id}): en az {count} sorgu"
            for layer_id, count in query_distribution.items()
        )
        layer_instructions = self._format_layer_instructions(query_distribution)

        prompt_messages = self.layered_query_prompt.format_messages(
            topic=topic,
            query_allocation=allocation_lines,
            layer_instructions=layer_instructions,
        )
        plan_prompt_text = next(
            (msg.content for msg in prompt_messages if isinstance(msg, HumanMessage)),
            "",
        )

        response = await self.llm.ainvoke(prompt_messages)
        raw_content = getattr(response, "content", str(response))

        normalized_plan: Optional[OrderedDict] = None
        parsed = None
        try:
            parsed = parse_json_from_response(raw_content)
        except Exception as exc:  # pragma: no cover - hata loglama
            logger.warning("Sorgu planı JSON formatına parse edilemedi: %s", exc)

        if parsed:
            normalized_plan = self._normalize_query_plan(topic, parsed, query_distribution)

        if not normalized_plan:
            logger.info("LLM çıktısı kullanılamadı, fallback sorgu planı oluşturuluyor.")
            normalized_plan = self._build_fallback_plan(topic, query_distribution)
            plan_response_text = json.dumps(
                {"layers": normalized_plan}, ensure_ascii=False, indent=2
            )
        else:
            plan_response_text = raw_content

        return normalized_plan, plan_prompt_text, plan_response_text

    async def _execute_query_plan(
        self,
        topic: str,
        query_plan: OrderedDict,
        *,
        variant: str,
    ) -> Tuple[OrderedDict, List[ToolMessage]]:
        """Sorgu planını çalıştır ve sonuçları döndür."""

        results: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict(
            (layer_id, []) for layer_id in query_plan.keys()
        )
        tool_messages: List[ToolMessage] = []

        for layer_id, layer_data in query_plan.items():
            queries = layer_data.get("queries", [])
            for index, query in enumerate(queries, 1):
                query_text = str(query).strip()
                if not query_text:
                    continue
                logger.info("Arama: [%s] %s", layer_id, query_text)
                normalized_topic = self._normalize_search_topic(topic)
                args = {"queries": [query_text]}
                if normalized_topic:
                    args["topic"] = normalized_topic
                try:
                    search_output = await self.search_tool.ainvoke(args)
                except Exception as exc:  # pragma: no cover - dış servis hataları
                    logger.error("Arama sırasında hata oluştu: %s", exc)
                    search_output = f"⚠️ Arama sırasında hata oluştu: {exc}"
                results[layer_id].append(
                    {
                        "layer": layer_id,
                        "layer_title": layer_data.get("title"),
                        "query": query_text,
                        "variant": variant,
                        "result": search_output,
                    }
                )
                tool_messages.append(
                    ToolMessage(
                        content=search_output,
                        tool_call_id=f"search_{variant}_{layer_id}_{index}",
                    )
                )
        return results, tool_messages

    async def _analyze_initial_findings(
        self,
        topic: str,
        query_plan: OrderedDict,
        initial_results: OrderedDict,
        quality_scores: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, str]:
        """İlk arama sonuçlarını analiz et."""

        query_plan_summary = self._format_query_plan_summary(query_plan)
        top_sources_summary = self._format_source_summary(quality_scores, limit=8)
        search_digest = self._build_search_digest(initial_results, max_chars=4500)

        prompt_messages = self.analysis_prompt.format_messages(
            topic=topic,
            query_plan_summary=query_plan_summary,
            top_sources=top_sources_summary or "Güçlü kaynak bulunamadı.",
            search_digest=search_digest or "Arama çıktısı bulunamadı.",
        )
        analysis_prompt_text = next(
            (msg.content for msg in prompt_messages if isinstance(msg, HumanMessage)),
            "",
        )

        response = await self.llm.ainvoke(prompt_messages)
        raw_content = getattr(response, "content", str(response))

        parsed = None
        try:
            parsed = parse_json_from_response(raw_content)
        except Exception as exc:  # pragma: no cover - hata loglama
            logger.warning("Analiz çıktısı parse edilemedi: %s", exc)

        if not parsed:
            parsed = self._default_analysis_structure()
            analysis_response_text = json.dumps(parsed, ensure_ascii=False, indent=2)
        else:
            analysis_response_text = raw_content

        analysis_data = self._normalize_analysis_output(parsed)
        return analysis_data, analysis_prompt_text, analysis_response_text

    async def _synthesize(
        self,
        topic: str,
        query_plan: OrderedDict,
        analysis_data: Dict[str, Any],
        follow_up_digest: str,
        combined_digest: str,
        top_sources_summary: str,
    ) -> Tuple[str, str]:
        """Tüm bulguları sentezle."""

        query_plan_summary = self._format_query_plan_summary(query_plan)
        analysis_json = json.dumps(analysis_data, ensure_ascii=False)

        prompt_messages = self.synthesis_prompt.format_messages(
            topic=topic,
            query_plan_summary=query_plan_summary,
            analysis_json=analysis_json[:4000],
            follow_up_digest=follow_up_digest or "Takip araması yapılmadı.",
            combined_digest=combined_digest or "Arama çıktısı bulunamadı.",
            top_sources=top_sources_summary or "Öne çıkan kaynak bulunamadı.",
        )
        synthesis_prompt_text = next(
            (msg.content for msg in prompt_messages if isinstance(msg, HumanMessage)),
            "",
        )

        response = await self.llm.ainvoke(prompt_messages)
        final_summary_text = getattr(response, "content", str(response))
        return final_summary_text, synthesis_prompt_text

    def _distribute_query_counts(self, total_requested: int) -> OrderedDict:
        """Toplam sorgu sayısını katmanlara dağıt."""

        total_requested = max(int(total_requested or 0), 1)
        layer_ids = list(self.research_layers.keys())
        total_layers = len(layer_ids)
        base = total_requested // total_layers

        if base == 0:
            return OrderedDict((layer_id, 1) for layer_id in layer_ids)

        distribution = OrderedDict((layer_id, base) for layer_id in layer_ids)
        remainder = total_requested - base * total_layers

        index = 0
        while remainder > 0:
            layer_id = layer_ids[index % total_layers]
            distribution[layer_id] += 1
            remainder -= 1
            index += 1
        return distribution

    def _format_layer_instructions(self, query_distribution: Dict[str, int]) -> str:
        """Katman talimatlarını metin haline getir."""

        lines: List[str] = []
        for layer_id, layer in self.research_layers.items():
            objectives = "\n".join(f"• {item}" for item in layer.get("objectives", []))
            sources = ", ".join(layer.get("source_targets", []))
            target = query_distribution.get(layer_id, 1)
            lines.append(
                f"- {layer['title']} ({layer_id})\n"
                f"  Odak: {objectives}\n"
                f"  Kaynak önceliği: {sources}\n"
                f"  Minimum sorgu: {target}"
            )
        return "\n".join(lines)

    def _normalize_query_plan(
        self,
        topic: str,
        raw_plan: Dict[str, Any],
        query_distribution: Dict[str, int],
    ) -> Optional[OrderedDict]:
        """LLM çıktısını normalize ederek kullanılabilir plana dönüştür."""

        if not isinstance(raw_plan, dict):
            return None

        layers_payload = raw_plan.get("layers", raw_plan)
        normalized = OrderedDict()

        for layer_id, layer in self.research_layers.items():
            layer_info = self._extract_layer_info(layers_payload, layer_id)
            focus = layer.get("description", "")
            if isinstance(layer_info, dict):
                focus = str(
                    layer_info.get("focus")
                    or layer_info.get("objective")
                    or layer_info.get("summary")
                    or focus
                )

            queries_raw = []
            source_targets = layer.get("source_targets", [])
            if isinstance(layer_info, dict):
                queries_raw = layer_info.get("queries") or layer_info.get("search_queries") or []
                raw_sources = layer_info.get("source_targets") or layer_info.get("sources")
                if isinstance(raw_sources, (list, tuple)):
                    source_targets = [str(item) for item in raw_sources if str(item).strip()]
            elif isinstance(layer_info, list):
                queries_raw = layer_info

            queries = self._ensure_query_count(
                topic,
                layer,
                queries_raw if isinstance(queries_raw, Iterable) else [],
                query_distribution.get(layer_id, 1),
            )

            normalized[layer_id] = {
                "title": layer["title"],
                "focus": focus,
                "objectives": layer.get("objectives", []),
                "source_targets": source_targets,
                "queries": queries,
            }

        return normalized

    def _build_fallback_plan(
        self, topic: str, query_distribution: Dict[str, int]
    ) -> OrderedDict:
        """Heuristik fallback sorgu planı üret."""

        fallback = OrderedDict()
        for layer_id, layer in self.research_layers.items():
            queries = self._ensure_query_count(
                topic,
                layer,
                [],
                query_distribution.get(layer_id, 1),
            )
            fallback[layer_id] = {
                "title": layer["title"],
                "focus": layer.get("description", ""),
                "objectives": layer.get("objectives", []),
                "source_targets": layer.get("source_targets", []),
                "queries": queries,
            }
        return fallback

    def _ensure_query_count(
        self,
        topic: str,
        layer: Dict[str, Any],
        queries_raw: Iterable,
        minimum: int,
    ) -> List[str]:
        """Sorgu listesini temizle ve gerekli sayıya tamamla."""

        cleaned: List[str] = []
        seen = set()
        for item in queries_raw or []:
            if isinstance(item, str):
                text = item.strip()
                if text and text.lower() not in seen:
                    cleaned.append(text)
                    seen.add(text.lower())

        seed_queries = [pattern.format(topic=topic) for pattern in layer.get("seed_queries", [])]
        for seed in seed_queries:
            if len(cleaned) >= minimum:
                break
            if seed.lower() not in seen:
                cleaned.append(seed)
                seen.add(seed.lower())

        index = 1
        while len(cleaned) < minimum:
            candidate = f"{topic} {layer['id']} derinlemesine araştırma {index}"
            if candidate.lower() not in seen:
                cleaned.append(candidate)
                seen.add(candidate.lower())
            index += 1

        return cleaned

    def _extract_layer_info(self, layers_payload: Any, layer_id: str) -> Any:
        """Katman bilgilerini esnek şekilde çıkar."""

        if isinstance(layers_payload, dict):
            for key, value in layers_payload.items():
                if isinstance(key, str):
                    key_lower = key.lower()
                    if key_lower == layer_id:
                        return value
                    if layer_id in key_lower:
                        return value
        return None

    def _merge_layer_results(self, *result_sets: Dict[str, List[Dict[str, Any]]]) -> OrderedDict:
        """Birden fazla sonuç setini birleştir."""

        merged = OrderedDict((layer_id, []) for layer_id in self.research_layers.keys())
        for result_set in result_sets:
            if not result_set:
                continue
            for layer_id, entries in result_set.items():
                merged.setdefault(layer_id, [])
                merged[layer_id].extend(entries or [])
        return merged

    def _build_search_digest(
        self, results: Dict[str, List[Dict[str, Any]]], *, max_chars: int
    ) -> str:
        """Arama sonuçlarını özetleyen metin üret."""

        if not results:
            return ""

        parts: List[str] = []
        for layer_id, entries in results.items():
            if not entries:
                continue
            layer_title = self.research_layers.get(layer_id, {}).get("title", layer_id)
            parts.append(f"### {layer_title} ({layer_id})")
            for entry in entries:
                query = entry.get("query", "")
                snippet = str(entry.get("result", "")).strip()
                if len(snippet) > 600:
                    snippet = snippet[:600] + "..."
                parts.append(f"- Sorgu: {query}\n{snippet}")

        digest = "\n".join(parts)
        if len(digest) > max_chars:
            digest = digest[:max_chars] + "..."
        return digest

    def _format_query_plan_summary(self, query_plan: OrderedDict) -> str:
        """Sorgu planını metinsel özet haline getir."""

        lines: List[str] = []
        for layer_id, info in query_plan.items():
            title = info.get("title", layer_id)
            queries = info.get("queries", [])
            lines.append(f"{title} ({layer_id}) -> {len(queries)} sorgu")
            for query in queries[:4]:
                lines.append(f"  - {query}")
        return "\n".join(lines)

    def _format_source_summary(
        self, quality_scores: List[Dict[str, Any]], *, limit: int
    ) -> str:
        """Kaynak kalitesi özetini hazırla."""

        if not quality_scores:
            return ""

        lines: List[str] = []
        for item in quality_scores[:limit]:
            signals = ", ".join(item.get("signals", [])[:3])
            layers = ", ".join(item.get("layers", []))
            lines.append(
                f"- [{item.get('quality_tier', '').upper()} | {item.get('score', 0)}] "
                f"{item.get('domain')} ({layers}) -> {item.get('url')}"
                + (f" | {signals}" if signals else "")
            )
        return "\n".join(lines)

    def _extract_urls(self, text: str) -> List[str]:
        """Arama çıktısından URL'leri çıkar."""

        urls = set()
        for match in URL_PATTERN.findall(text or ""):
            cleaned = match.strip().rstrip(").,;")
            if cleaned:
                urls.add(cleaned)
        return list(urls)

    def _evaluate_url(self, url: str) -> Optional[Dict[str, Any]]:
        """URL için kalite sinyalleri hesapla."""

        if not url:
            return None
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not domain:
            return None

        score = 1 if parsed.scheme == "https" else 0
        signals: List[str] = []
        source_type = "general"
        highest_priority = 0
        url_lower = url.lower()

        for rule in SOURCE_SIGNAL_RULES:
            for pattern in rule["patterns"]:
                target = domain if pattern.startswith(".") else url_lower
                if pattern in target:
                    score += rule["score"]
                    signals.append(rule["signal"])
                    priority = TYPE_PRIORITY.get(rule["type"], 0)
                    if priority >= highest_priority:
                        source_type = rule["type"]
                        highest_priority = priority
                    break

        if domain.endswith(".org") and "STK" not in signals:
            signals.append("STK/.org alan adı")
            score += 1

        score = min(score, 10)
        tier = "high" if score >= 7 else "medium" if score >= 4 else "low"

        return {
            "url": url,
            "domain": domain,
            "score": score,
            "quality_tier": tier,
            "signals": signals,
            "source_type": source_type,
        }

    def _score_sources(
        self, results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Arama sonuçlarından kaynak kalitesi puanları üret."""

        aggregated: Dict[str, Dict[str, Any]] = {}

        for layer_id, entries in results.items():
            for entry in entries:
                query = entry.get("query", "")
                for url in self._extract_urls(entry.get("result", "")):
                    evaluation = self._evaluate_url(url)
                    if not evaluation:
                        continue
                    record = aggregated.setdefault(
                        evaluation["url"],
                        {
                            "url": evaluation["url"],
                            "domain": evaluation["domain"],
                            "score": evaluation["score"],
                            "quality_tier": evaluation["quality_tier"],
                            "source_type": evaluation["source_type"],
                            "signals": set(evaluation.get("signals", [])),
                            "layers": set(),
                            "queries": set(),
                        },
                    )
                    record["score"] = max(record["score"], evaluation["score"])
                    record["quality_tier"] = (
                        "high"
                        if evaluation["score"] >= 7 or record["score"] >= 7
                        else "medium"
                        if evaluation["score"] >= 4 or record["score"] >= 4
                        else record["quality_tier"]
                    )
                    record["source_type"] = evaluation["source_type"]
                    record["signals"].update(evaluation.get("signals", []))
                    record["layers"].add(layer_id)
                    if query:
                        record["queries"].add(query)

        scored_list: List[Dict[str, Any]] = []
        for data in aggregated.values():
            scored_list.append(
                {
                    "url": data["url"],
                    "domain": data["domain"],
                    "score": data["score"],
                    "quality_tier": data["quality_tier"],
                    "source_type": data["source_type"],
                    "signals": sorted(data["signals"]),
                    "layers": sorted(data["layers"]),
                    "queries": sorted(data["queries"]),
                }
            )

        scored_list.sort(key=lambda item: (-item["score"], item["domain"]))
        return scored_list

    def _default_analysis_structure(self) -> Dict[str, Any]:
        """Analiz çıktısı için varsayılan yapı."""

        return {
            "patterns": [],
            "contradictions": [],
            "trends": [],
            "gaps": OrderedDict(
                (
                    layer_id,
                    {
                        "missing_topics": [],
                        "follow_up_queries": [],
                        "validation_targets": [],
                    },
                )
                for layer_id in self.research_layers.keys()
            ),
            "cross_validation": [],
        }

    def _normalize_analysis_output(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Analiz çıktısını normalize et."""

        normalized = self._default_analysis_structure()
        if not isinstance(raw, dict):
            return normalized

        if isinstance(raw.get("patterns"), list):
            normalized["patterns"] = raw["patterns"]
        if isinstance(raw.get("contradictions"), list):
            normalized["contradictions"] = raw["contradictions"]
        if isinstance(raw.get("trends"), list):
            normalized["trends"] = raw["trends"]
        if isinstance(raw.get("cross_validation"), list):
            normalized["cross_validation"] = raw["cross_validation"]

        gaps_raw = raw.get("gaps")
        if isinstance(gaps_raw, dict):
            for key, value in gaps_raw.items():
                layer_id = self._match_layer_id(key)
                if not layer_id or not isinstance(value, dict):
                    continue
                normalized["gaps"][layer_id] = {
                    "missing_topics": self._coerce_str_list(
                        value.get("missing_topics") or value.get("gaps")
                    ),
                    "follow_up_queries": self._coerce_str_list(
                        value.get("follow_up_queries") or value.get("queries")
                    ),
                    "validation_targets": self._coerce_str_list(
                        value.get("validation_targets") or value.get("validation")
                    ),
                }
        elif isinstance(gaps_raw, list):
            for item in gaps_raw:
                if not isinstance(item, dict):
                    continue
                layer_id = self._match_layer_id(item.get("layer") or item.get("layer_id"))
                if not layer_id:
                    continue
                normalized["gaps"][layer_id] = {
                    "missing_topics": self._coerce_str_list(
                        item.get("missing_topics") or item.get("topics")
                    ),
                    "follow_up_queries": self._coerce_str_list(
                        item.get("follow_up_queries") or item.get("queries")
                    ),
                    "validation_targets": self._coerce_str_list(
                        item.get("validation_targets") or item.get("validation")
                    ),
                }

        return normalized

    def _match_layer_id(self, raw_value: Any) -> Optional[str]:
        """Serbest metni katman kimliği ile eşleştir."""

        if raw_value is None:
            return None
        text = str(raw_value).lower()
        for layer_id, layer in self.research_layers.items():
            if text == layer_id:
                return layer_id
            for alias in layer.get("aliases", []):
                if alias and alias.lower() in text:
                    return layer_id
        return None

    def _coerce_str_list(self, value: Any) -> List[str]:
        """Değeri str listesine dönüştür."""

        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    result.append(item.strip())
            return result
        return []

    def _collect_follow_up_queries(
        self, topic: str, analysis_data: Dict[str, Any]
    ) -> OrderedDict:
        """Analiz çıktısından takip sorgularını çıkar."""

        follow_up = OrderedDict()
        if not isinstance(analysis_data, dict):
            return follow_up

        gaps = analysis_data.get("gaps") or {}
        for layer_id, layer in self.research_layers.items():
            layer_gap: Optional[Dict[str, Any]] = None
            if isinstance(gaps, dict):
                for key, value in gaps.items():
                    matched = self._match_layer_id(key)
                    if matched == layer_id and isinstance(value, dict):
                        layer_gap = value
                        break
            elif isinstance(gaps, list):
                for item in gaps:
                    if not isinstance(item, dict):
                        continue
                    matched = self._match_layer_id(item.get("layer") or item.get("layer_id"))
                    if matched == layer_id:
                        layer_gap = item
                        break

            if not isinstance(layer_gap, dict):
                continue

            queries = self._coerce_str_list(
                layer_gap.get("follow_up_queries") or layer_gap.get("queries")
            )
            if not queries:
                missing_topics = self._coerce_str_list(layer_gap.get("missing_topics"))
                for topic_name in missing_topics[:2]:
                    candidate = f"{topic} {topic_name} derinlemesine araştırma"
                    if candidate not in queries:
                        queries.append(candidate)

            queries = [query for query in queries if query]
            if queries:
                follow_up[layer_id] = queries[:3]

        return follow_up


# Test fonksiyonu
async def test_researcher():
    """Araştırmacı ajanını test et."""

    from report_agent_setup import create_llm, search_web

    llm = create_llm()
    researcher = ResearcherAgent(llm, search_web)

    result = await researcher.research(
        topic="Yapay zeka ajanlarının sağlık sektöründeki uygulamaları",
        number_of_queries=5,
    )

    print("\n=== KATMANLI ARAŞTIRMA SONUCU ===")
    print(result.get("final_summary", "Sonuç bulunamadı")[:1200])

    print("\n=== ÖNE ÇIKAN KAYNAKLAR ===")
    for item in result.get("quality_scores", [])[:5]:
        print(
            f"- {item['domain']} | skor={item['score']} | tür={item['source_type']}\n  {item['url']}"
        )

    print("\n=== TAKİP SORGULARI ===")
    for layer_id, queries in result.get("follow_up_queries", {}).items():
        print(f"{layer_id}: {queries}")


if __name__ == "__main__":
    asyncio.run(test_researcher())
