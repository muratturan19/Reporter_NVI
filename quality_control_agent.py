# -*- coding: utf-8 -*-
"""
Report Quality Control Agent - Rapor kalitesini kontrol eden ve düzelten ajan
"""

import re
import logging
from typing import Dict, List, Tuple
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class ReportQualityAgent:
    """Rapor kalitesini kontrol eden ve düzelten ajan"""

    def __init__(self, llm):
        self.llm = llm

        # Quality check prompts
        self.quality_check_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen bir rapor kalite kontrol uzmanısın. Verilen raporu incele ve aşağıdaki sorunları tespit et:

1. ENCODING SORUNLARI:
   - Bozuk UTF-8 karakterler
   - Yabancı dil karakterleri (Arapça, vb.)
   - Garbled text

2. MARKDOWN/FORMAT SORUNLARI:
   - Broken HTML linkler
   - Bozuk başlık formatları
   - Eksik veya yanlış markdown syntax

3. İÇERİK SORUNLARI:
   - Anlamsız kelimeler
   - Yarım kalmış cümleler
   - Dil karışımları (Türkçe-İngilizce karışımı)
   - Tekrarlanan veya contradictory bilgiler

4. YAPISAL SORUNLAR:
   - Eksik bölümler
   - Broken navigation
   - Inconsistent formatting

Sadece sorunları listele, düzeltme yapma. JSON formatında yanıt ver:
{
  "encoding_issues": ["sorun1", "sorun2"],
  "format_issues": ["sorun1", "sorun2"], 
  "content_issues": ["sorun1", "sorun2"],
  "structural_issues": ["sorun1", "sorun2"],
  "severity": "low|medium|high",
  "overall_score": 0-100
}"""),
            ("human", "Rapor:\n\n{report}")
        ])

        self.fix_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen bir rapor düzeltme uzmanısın. Verilen rapordaki sorunları düzelt:

DÜZELTME KURALLARI:
1. Bozuk UTF-8 karakterleri temizle veya düzelt
2. Yabancı dil karakterlerini kaldır  
3. Broken HTML linklerini düzelt
4. Markdown formatını onar
5. Anlamsız kelimeleri mantıklı kelimelerle değiştir
6. Yarım cümleleri tamamla veya kaldır
7. Dil tutarlılığını sağla (sadece Türkçe)
8. Başlık hiyerarşisini düzelt
9. Navigation linklerini onar
10. İçerik akışını mantıklı hale getir

SADECE düzeltilmiş raporu dön. Hiçbir ek açıklama yapma."""),
            ("human", """Düzeltilecek rapor:

{report}

Tespit edilen sorunlar:
{issues}

Düzeltilmiş raporu ver:""")
        ])

    def detect_encoding_issues(self, text: str) -> List[str]:
        """Encoding sorunlarını tespit et"""
        issues = []

        # Yaygın encoding sorunları
        encoding_patterns = [
            (r'[^\x00-\x7F\u00A0-\u017F\u0100-\u024F\u1E00-\u1EFF]', 'Non-Latin characters detected'),
            (r'Ã[€-Â]', 'UTF-8 encoding corruption'),
            (r'â€™|â€œ|â€\x9d', 'Smart quotes encoding issue'),
            (r'Ä±|Åž|Ä\x9f|Ã§|Ã¼|Ã¶', 'Turkish character encoding issue'),
        ]

        for pattern, description in encoding_patterns:
            if re.search(pattern, text):
                issues.append(description)

        return issues

    def detect_format_issues(self, text: str) -> List[str]:
        """Format sorunlarını tespit et"""
        issues = []

        # Markdown format sorunları
        if re.search(r'##[^#\s]', text):
            issues.append('Broken heading format detected')

        if re.search(r'\]\([^)]*$', text):
            issues.append('Unclosed markdown links')

        if re.search(r'<a name="[^"]*">[^<]*</a>', text):
            issues.append('HTML anchor tags in markdown')

        if re.search(r'###\s*$', text):
            issues.append('Empty headings detected')

        return issues

    def detect_content_issues(self, text: str) -> List[str]:
        """İçerik sorunlarını tespit et"""
        issues = []

        # Anlamsız kelimeler (örnekler)
        nonsense_words = [
            'Kalıtschaft', 'Sirküt', 'DavyBinary', 'Outre', 'famoso',
            'Napıлий', 'oscillations', 'کاکma', 'گ', 'الفقر'
        ]

        for word in nonsense_words:
            if word in text:
                issues.append(f'Nonsense word detected: {word}')

        # Yarım cümleler
        if re.search(r'\w+\?\s*$', text, re.MULTILINE):
            issues.append('Incomplete sentences ending with ?')

        # Dil karışımları
        if re.search(r'[a-zA-Z]{3,}\s+[^\x00-\x7F]+', text):
            issues.append('Mixed language content detected')

        return issues

    def detect_structural_issues(self, text: str) -> List[str]:
        """Yapısal sorunları tespit et"""
        issues = []

        # Broken navigation
        if re.search(r'##[^#].*\[.*\]\(##.*\)', text):
            issues.append('Broken internal navigation links')

        # Inconsistent heading levels
        headings = re.findall(r'^(#{1,6})\s', text, re.MULTILINE)
        if headings:
            levels = [len(h) for h in headings]
            if max(levels) - min(levels) > 3:
                issues.append('Inconsistent heading hierarchy')

        return issues

    async def analyze_quality(self, report: str) -> Dict:
        """Rapor kalitesini analiz et"""
        try:
            messages = self.quality_check_prompt.format_messages(report=report)
            response = await self.llm.ainvoke(messages)

            # JSON parsing
            from json_parser_fix import parse_json_from_response

            try:
                quality_data = parse_json_from_response(response.content)
                return quality_data
            except Exception as e:
                logger.warning(f"JSON parsing failed, using fallback analysis: {e}")
                return self._fallback_analysis(report)

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return self._fallback_analysis(report)

    def _fallback_analysis(self, report: str) -> Dict:
        """LLM çalışmazsa manual analysis"""
        encoding_issues = self.detect_encoding_issues(report)
        format_issues = self.detect_format_issues(report)
        content_issues = self.detect_content_issues(report)
        structural_issues = self.detect_structural_issues(report)

        total_issues = len(encoding_issues) + len(format_issues) + len(content_issues) + len(structural_issues)

        if total_issues >= 10:
            severity = "high"
            score = 20
        elif total_issues >= 5:
            severity = "medium"
            score = 50
        else:
            severity = "low"
            score = 80

        return {
            "encoding_issues": encoding_issues,
            "format_issues": format_issues,
            "content_issues": content_issues,
            "structural_issues": structural_issues,
            "severity": severity,
            "overall_score": score
        }

    async def fix_report(self, report: str, issues: Dict) -> str:
        """Raporu düzelt"""
        try:
            # İlk önce basic cleaning yap
            cleaned_report = self._basic_cleanup(report)

            # Eğer sorunlar çok fazlaysa LLM ile düzeltmeyi dene
            if issues.get("overall_score", 100) < 60:
                logger.info("Running LLM-based report fixing...")

                issues_summary = []
                for category, problems in issues.items():
                    if isinstance(problems, list) and problems:
                        issues_summary.extend(problems)

                issues_text = "\n".join(f"- {issue}" for issue in issues_summary[:10])

                messages = self.fix_prompt.format_messages(
                    report=cleaned_report,
                    issues=issues_text
                )

                response = await self.llm.ainvoke(messages)
                fixed_report = response.content.strip()

                # Basic validation
                if len(fixed_report) < len(cleaned_report) * 0.5:
                    logger.warning("Fixed report too short, using cleaned version")
                    return cleaned_report

                return fixed_report
            else:
                return cleaned_report

        except Exception as e:
            logger.error(f"Report fixing failed: {e}")
            return self._basic_cleanup(report)

    def _basic_cleanup(self, text: str) -> str:
        """Temel temizlik işlemleri"""
        # Non-printable karakterleri temizle
        text = re.sub(r'[^\x20-\x7E\u00A0-\u017F\u0100-\u024F\u1E00-\u1EFF]', '', text)

        # Broken HTML anchor tags düzelt
        text = re.sub(r'<a name="([^"]*)"[^>]*>([^<]*)</a>', r'## \2', text)

        # Broken heading links düzelt  
        text = re.sub(r'##([^#]*)\[.*\]\(##([^)]*)\)', r'## \1', text)

        # Multiple spaces düzelt
        text = re.sub(r' {2,}', ' ', text)

        # Empty lines düzelt
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Broken markdown links temizle
        text = re.sub(r'\]\([^)]*$', '', text)

        return text.strip()

    async def process_report(self, report: str) -> Tuple[str, Dict]:
        """Ana işlem: analiz et ve düzelt"""
        logger.info("Quality control started...")

        # Kalite analizi
        quality_analysis = await self.analyze_quality(report)

        logger.info(f"Quality score: {quality_analysis.get('overall_score', 'unknown')}")
        logger.info(f"Severity: {quality_analysis.get('severity', 'unknown')}")

        # Düzeltme
        if quality_analysis.get("overall_score", 100) < 70:
            logger.info("Fixing report...")
            fixed_report = await self.fix_report(report, quality_analysis)
        else:
            logger.info("Report quality acceptable, applying basic cleanup only")
            fixed_report = self._basic_cleanup(report)

        logger.info("Quality control completed")

        return fixed_report, quality_analysis
