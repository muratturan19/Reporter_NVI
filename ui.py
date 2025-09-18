# -*- coding: utf-8 -*-
"""Gradio tabanlı kullanıcı arayüzü.

Bu arayüz, mevcut rapor oluşturma ajanını kullanarak kullanıcıların konu
başlıklarını girmesine ve oluşturulan raporu hem görüntülemesine hem de
indirmesine olanak tanır.
"""

from __future__ import annotations

import inspect
import os
from datetime import datetime
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

from main_report_agent import MainReportAgent

# Ortam değişkenlerini yükle (API anahtarları vb.)
load_dotenv()


_agent: Optional[MainReportAgent] = None


def _get_agent() -> MainReportAgent:
    """MainReportAgent örneğini tekil olacak şekilde döndür."""

    global _agent
    if _agent is None:
        _agent = MainReportAgent()
    return _agent


def _sanitize_topic(topic: str) -> str:
    """Dosya adı için konu başlığını güvenli formata getir."""

    safe_chars = [ch if ch.isalnum() else " " for ch in topic.lower()]
    sanitized = "_".join("".join(safe_chars).split())
    if not sanitized:
        sanitized = "rapor"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sanitized[:50]}_{timestamp}.md"


async def run_report(topic: str, progress=gr.Progress(track_tqdm=True)):
    """Gradio arayüzü için rapor oluştur."""

    cleaned_topic = (topic or "").strip()
    if not cleaned_topic:
        warning = "❌ Lütfen bir rapor konusu girin."
        return (
            gr.update(value=""),
            gr.update(value=None, visible=False),
            gr.update(value=warning),
        )

    progress(0.0, desc="Ajan hazırlanıyor...")
    agent = _get_agent()

    progress(0.1, desc="Rapor oluşturuluyor. Bu işlem birkaç dakika sürebilir...")
    report = await agent.generate_report(cleaned_topic)

    if not report:
        progress(1.0, desc="Hata oluştu")
        message = "❌ Rapor oluşturulamadı. Lütfen tekrar deneyin."
        return (
            gr.update(value=""),
            gr.update(value=None, visible=False),
            gr.update(value=message),
        )

    normalized = report.strip().lower()
    lines = normalized.splitlines()
    first_line = lines[0] if lines else normalized
    if normalized.startswith("hata:") or first_line.startswith("hata"):
        progress(1.0, desc="Hata oluştu")
        # Hata mesajlarını kullanıcıya aktar.
        return (
            gr.update(value=report),
            gr.update(value=None, visible=False),
            gr.update(value=f"❌ {report}"),
        )

    progress(0.9, desc="Rapor kaydediliyor...")
    filename = _sanitize_topic(cleaned_topic)
    filepath = await agent.save_report(report, filename=filename)

    if filepath:
        status_text = (
            f"✅ Rapor hazır! İndirilebilir dosya: {os.path.basename(filepath)}"
        )
        file_output = gr.update(value=filepath, visible=True)
    else:
        status_text = (
            "⚠️ Rapor oluşturuldu ancak dosya kaydedilemedi."
            " Raporu metin alanından kopyalayabilirsiniz."
        )
        file_output = gr.update(value=None, visible=False)

    progress(1.0, desc="Tamamlandı")

    return (
        gr.update(value=report),
        file_output,
        gr.update(value=status_text),
    )


def build_interface() -> gr.Blocks:
    """Gradio Blocks arayüzünü kur."""

    with gr.Blocks(title="NVIDIA Rapor Ajanı") as demo:
        gr.Markdown(
            """
            # 🧠 NVIDIA Rapor Ajanı

            Bir konu başlığı girin, sistem web'de araştırma yaparak kapsamlı bir
            Markdown raporu oluştursun. `.env` dosyanızda gerekli API anahtarları
            bulunduğundan emin olun.
            """
        )

        topic_input = gr.Textbox(
            label="Rapor Konusu",
            placeholder="Örn. 'Yapay zeka ajanlarının sağlık sektöründeki uygulamaları'",
        )

        generate_button = gr.Button("Rapor Oluştur", variant="primary")

        status_box = gr.Markdown("", elem_id="status-box")
        report_output = gr.Markdown(label="Oluşturulan Rapor")
        download_output = gr.File(
            label="Raporu indir",
            visible=False,
        )

        gr.Examples(
            [
                "Yapay zeka destekli müşteri hizmetleri çözümleri",
                "Sürdürülebilir enerji yönetiminde dijital ikiz uygulamaları",
                "Finans sektöründe büyük dil modellerinin kullanımı",
            ],
            inputs=topic_input,
        )

        generate_button.click(
            run_report,
            inputs=topic_input,
            outputs=[report_output, download_output, status_box],
        )

        topic_input.submit(
            run_report,
            inputs=topic_input,
            outputs=[report_output, download_output, status_box],
        )

    return demo


def launch():
    """Arayüzü başlat."""

    demo = build_interface()

    queue_kwargs = {}
    queue_params = inspect.signature(gr.Blocks.queue).parameters

    if "default_concurrency_limit" in queue_params:
        queue_kwargs["default_concurrency_limit"] = 1
    elif "concurrency_count" in queue_params:
        queue_kwargs["concurrency_count"] = 1

    demo.queue(**queue_kwargs).launch()


if __name__ == "__main__":
    launch()
