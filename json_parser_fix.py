# -*- coding: utf-8 -*-
"""JSON Parser düzeltmesi - plan_report fonksiyonunda kullanılacak"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def _find_last_unescaped_quote(segment: str) -> int | None:
    """Return index of the last unescaped double quote in the given segment."""

    idx = len(segment) - 1
    while idx >= 0:
        if segment[idx] == '"':
            backslash_count = 0
            lookback = idx - 1
            while lookback >= 0 and segment[lookback] == '\\':
                backslash_count += 1
                lookback -= 1
            if backslash_count % 2 == 0:
                return idx
        idx -= 1
    return None


def _escape_unescaped_quotes(value: str) -> tuple[str, bool]:
    """Escape naked double quotes inside a JSON string value."""

    escaped_chars = []
    backslash_run = 0
    modified = False

    for char in value:
        if char == '\\':
            escaped_chars.append(char)
            backslash_run += 1
            continue

        if char == '"':
            if backslash_run % 2 == 0:
                escaped_chars.append('\\"')
                modified = True
            else:
                escaped_chars.append(char)
            backslash_run = 0
            continue

        escaped_chars.append(char)
        backslash_run = 0

    return ''.join(escaped_chars), modified


def _normalize_unescaped_quotes(content: str) -> str:
    """Normalize unescaped quotes in string values without touching JSON structure."""

    field_pattern = re.compile(r'(^\s*"[^"\\]+"\s*:\s*")', re.MULTILINE)
    normalized_chunks = []
    cursor = 0
    content_changed = False

    for match in field_pattern.finditer(content):
        prefix_end = match.end()
        normalized_chunks.append(content[cursor:prefix_end])

        line_end = content.find('\n', prefix_end)
        if line_end == -1:
            line_end = len(content)

        segment = content[prefix_end:line_end]
        closing_rel = _find_last_unescaped_quote(segment)

        if closing_rel is None:
            normalized_chunks.append(segment)
            cursor = line_end
            continue

        value = segment[:closing_rel]
        fixed_value, modified = _escape_unescaped_quotes(value)

        if modified:
            content_changed = True
        normalized_chunks.append(fixed_value)
        normalized_chunks.append(segment[closing_rel:])
        cursor = line_end

    normalized_chunks.append(content[cursor:])

    if content_changed:
        logger.info("Model yanıtında kaçışsız çift tırnaklar düzeltildi.")

    return ''.join(normalized_chunks)


def parse_json_from_response(response_content: str) -> dict:
    """Model yanıtından JSON'ı güvenli şekilde çıkarır"""

    try:
        content = response_content.strip()
        logger.info(f"Ham model yanıtı (ilk 300 karakter): {content[:300]}")

        content = _normalize_unescaped_quotes(content)

        # Metod 1: Doğrudan JSON parse etmeyi dene
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Metod 2: İlk { ile son } arasını al
        json_start = content.find('{')
        json_end = content.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            logger.info(f"Çıkarılan JSON: {json_str}")
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Metod 3: Regex ile JSON bloğunu bul
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                # Çok satırlı JSON'ları temizle
                cleaned_match = re.sub(r'\n\s*', ' ', match)
                return json.loads(cleaned_match)
            except json.JSONDecodeError:
                continue

        # Metod 4: JSON'ı satır satır temizle
        lines = content.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0

        for line in lines:
            stripped = line.strip()
            if not in_json and stripped.startswith('{'):
                in_json = True
                json_lines.append(stripped)
                brace_count += stripped.count('{') - stripped.count('}')
            elif in_json:
                json_lines.append(stripped)
                brace_count += stripped.count('{') - stripped.count('}')
                if brace_count <= 0:
                    break

        if json_lines:
            json_str = ' '.join(json_lines)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Hiçbir metod çalışmadı
        raise ValueError("JSON formatı hiçbir metod ile parse edilemedi")

    except Exception as e:
        logger.error(f"JSON parsing hatası: {e}")
        logger.error(f"İşlenmeye çalışılan içerik: {response_content}")
        raise


def create_fallback_structure(topic: str) -> dict:
    """Fallback rapor yapısı oluştur"""
    return {
        "title": f"{topic} - Detaylı Araştırma Raporu",
        "sections": [
            {
                "name": "Giriş ve Kapsam",
                "description": f"{topic} konusunun tanıtımı, araştırmanın kapsamı ve amaçları",
                "research": False
            },
            {
                "name": "Mevcut Durum Analizi",
                "description": f"{topic} alanındaki mevcut durum, temel kavramlar ve güncel gelişmeler",
                "research": True
            },
            {
                "name": "Teknoloji ve Yöntemler",
                "description": f"{topic} kapsamında kullanılan teknolojiler, yöntemler ve araçlar",
                "research": True
            },
            {
                "name": "Uygulama Alanları",
                "description": f"{topic} konusunun pratik uygulama alanları ve gerçek dünya örnekleri",
                "research": True
            },
            {
                "name": "Fırsatlar ve Zorluklar",
                "description": f"{topic} alanındaki fırsatlar, zorluklar ve çözüm önerileri",
                "research": True
            },
            {
                "name": "Sonuç ve Öneriler",
                "description": "Araştırma bulgularının özeti, sonuçlar ve gelecek için öneriler",
                "research": False
            }
        ]
    }
