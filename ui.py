# -*- coding: utf-8 -*-
"""Geliştirilmiş Gradio tabanlı kullanıcı arayüzü.

Bu arayüz, mevcut rapor oluşturma ajanını kullanarak kullanıcıların konu
başlıklarını girmesine ve oluşturulan raporu hem görüntülemesine hem de
indirmesine olanak tanır. Real-time progress tracking ve modern görsel tasarım içerir.
"""

from __future__ import annotations

import inspect
import os
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional
from queue import Queue
import threading

import gradio as gr
from dotenv import load_dotenv

from main_report_agent import MainReportAgent

# Ortam değişkenlerini yükle
load_dotenv()

# Global değişkenler
_agent: Optional[MainReportAgent] = None
_log_queue = Queue()

# Custom CSS
CUSTOM_CSS = """
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 30px;
    color: white;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.status-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.progress-step {
    display: flex;
    align-items: center;
    padding: 10px;
    margin: 5px 0;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.step-waiting {
    background: #f1f5f9;
    color: #64748b;
}

.step-active {
    background: #dbeafe;
    color: #1d4ed8;
    border-left: 4px solid #3b82f6;
}

.step-completed {
    background: #dcfce7;
    color: #166534;
    border-left: 4px solid #22c55e;
}

.step-error {
    background: #fef2f2;
    color: #dc2626;
    border-left: 4px solid #ef4444;
}

.report-section {
    background: white;
    border-radius: 12px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border: 1px solid #e5e7eb;
}

.download-section {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
}

.error-message {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 15px;
    color: #dc2626;
}

.success-message {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 15px;
    color: #166534;
}
"""

class LogCapture(logging.Handler):
    """Custom logging handler to capture logs for UI"""
    
    def emit(self, record):
        log_entry = self.format(record)
        _log_queue.put({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'level': record.levelname,
            'message': log_entry,
            'name': record.name
        })

def setup_logging():
    """Setup logging to capture system logs"""
    handler = LogCapture()
    handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Ana logger'lara handler ekle
    loggers = [
        'main_report_agent',
        'researcher_agent', 
        'writer_agent',
        'json_parser_fix'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

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

def create_progress_steps():
    """Progress adımlarını oluştur"""
    steps = [
        {"id": "init", "text": "🚀 Sistem hazırlanıyor", "status": "waiting"},
        {"id": "research", "text": "🔍 Web araştırması yapılıyor", "status": "waiting"},
        {"id": "planning", "text": "📋 Rapor yapısı planlanıyor", "status": "waiting"},
        {"id": "writing", "text": "✍️ Bölümler yazılıyor", "status": "waiting"},
        {"id": "compiling", "text": "⚙️ Final rapor derleniyor", "status": "waiting"},
        {"id": "saving", "text": "💾 Rapor kaydediliyor", "status": "waiting"}
    ]
    return steps

def update_progress_display(steps):
    """Progress adımlarını HTML formatında döndür"""
    html = '<div class="status-card">'
    html += '<h3>📊 İşlem Durumu</h3>'
    
    for step in steps:
        status_class = f"step-{step['status']}"
        icon = {
            'waiting': '⏳',
            'active': '⚡',
            'completed': '✅',
            'error': '❌'
        }.get(step['status'], '⏳')
        
        html += f'''
        <div class="progress-step {status_class}">
            <span style="margin-right: 10px;">{icon}</span>
            <span>{step["text"]}</span>
        </div>
        '''
    
    html += '</div>'
    return html

def get_recent_logs(max_logs=10):
    """Son logları al ve formatla"""
    logs = []
    temp_logs = []
    
    # Queue'dan tüm logları al
    while not _log_queue.empty():
        temp_logs.append(_log_queue.get())
    
    # Son N log'u al
    recent_logs = temp_logs[-max_logs:] if temp_logs else []
    
    if not recent_logs:
        return "Henüz log yok..."
    
    html = '<div class="status-card">'
    html += '<h3>📜 Sistem Logları</h3>'
    html += '<div style="max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px;">'
    
    for log in recent_logs:
        color = {
            'INFO': '#22c55e',
            'WARNING': '#f59e0b', 
            'ERROR': '#ef4444',
            'DEBUG': '#6b7280'
        }.get(log['level'], '#6b7280')
        
        html += f'''
        <div style="margin: 5px 0; padding: 8px; background: #f8fafc; border-radius: 4px; border-left: 3px solid {color};">
            <span style="color: #64748b;">[{log["timestamp"]}]</span>
            <span style="color: {color}; font-weight: bold;">{log["level"]}</span>
            <span style="color: #334155;">: {log["message"]}</span>
        </div>
        '''
    
    html += '</div></div>'
    return html

async def run_report(topic: str):
    """Gelişmiş rapor oluşturma fonksiyonu"""
    cleaned_topic = (topic or "").strip()
    if not cleaned_topic:
        yield (
            "❌ Lütfen bir rapor konusu girin.",
            "",
            None,
            update_progress_display(create_progress_steps()),
            get_recent_logs()
        )
        return

    # Progress steps başlat
    steps = create_progress_steps()
    
    # Adım 1: Başlatma
    steps[0]["status"] = "active"
    yield (
        "🚀 Sistem başlatılıyor...",
        "",
        None,
        update_progress_display(steps),
        get_recent_logs()
    )
    
    try:
        agent = _get_agent()
        steps[0]["status"] = "completed"
        
        # Adım 2: Araştırma
        steps[1]["status"] = "active"
        yield (
            "🔍 Web araştırması başladı...",
            "",
            None,
            update_progress_display(steps),
            get_recent_logs()
        )
        
        # Rapor oluşturma - log takibi için
        start_time = time.time()
        report = await agent.generate_report(cleaned_topic)
        
        if not report or report.strip().lower().startswith("hata"):
            # Hata durumu
            for step in steps:
                if step["status"] == "active":
                    step["status"] = "error"
            
            error_msg = report if report else "Rapor oluşturulamadı. Lütfen tekrar deneyin."
            yield (
                f"❌ {error_msg}",
                "",
                None,
                update_progress_display(steps),
                get_recent_logs()
            )
            return
        
        # Tüm adımları tamamlandı olarak işaretle
        for step in steps[:-1]:  # Son adım hariç
            step["status"] = "completed"
        
        steps[-1]["status"] = "active"  # Kaydetme adımı
        yield (
            "💾 Rapor kaydediliyor...",
            report,
            None,
            update_progress_display(steps),
            get_recent_logs()
        )
        
        # Dosya kaydetme
        filename = _sanitize_topic(cleaned_topic)
        filepath = await agent.save_report(report, filename=filename)
        
        steps[-1]["status"] = "completed"
        
        # Başarılı tamamlama
        elapsed_time = time.time() - start_time
        success_message = f"✅ Rapor başarıyla oluşturuldu! ({elapsed_time:.1f} saniye)"
        
        if filepath:
            success_message += f"\n📁 Dosya: {os.path.basename(filepath)}"
        
        yield (
            success_message,
            report,
            filepath if filepath else None,
            update_progress_display(steps),
            get_recent_logs()
        )
        
    except Exception as e:
        # Genel hata yakalama
        for step in steps:
            if step["status"] == "active":
                step["status"] = "error"
        
        error_message = f"❌ Beklenmeyen hata: {str(e)}"
        yield (
            error_message,
            "",
            None,
            update_progress_display(steps),
            get_recent_logs()
        )

def build_interface() -> gr.Blocks:
    """Gelişmiş Gradio arayüzü oluştur"""
    
    with gr.Blocks(title="NVIDIA Rapor Ajanı", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        
        # Ana başlık
        gr.HTML("""
        <div class="header-section">
            <h1>🧠 NVIDIA Rapor Ajanı</h1>
            <p>Yapay zeka destekli araştırma ve rapor oluşturma sistemi</p>
            <p><em>Web araştırması yaparak kapsamlı Markdown raporları oluşturur</em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Giriş bölümü
                with gr.Group():
                    topic_input = gr.Textbox(
                        label="📝 Rapor Konusu",
                        placeholder="Örn. 'Yapay zeka ajanlarının sağlık sektöründeki uygulamaları'",
                        lines=2,
                        max_lines=3
                    )
                    
                    generate_button = gr.Button(
                        "🚀 Rapor Oluştur", 
                        variant="primary", 
                        size="lg"
                    )
                
                # Örnek konular
                gr.Examples(
                    label="💡 Örnek Konular",
                    examples=[
                        "Yapay zeka destekli müşteri hizmetleri çözümleri",
                        "Sürdürülebilir enerji yönetiminde dijital ikiz uygulamaları", 
                        "Finans sektöründe büyük dil modellerinin kullanımı",
                        "Endüstri 4.0 ve IoT sensörlerin üretim optimizasyonu",
                        "Blockchain teknolojisinin tedarik zinciri yönetimindeki rolü"
                    ],
                    inputs=topic_input
                )
            
            with gr.Column(scale=1):
                # Durum takibi
                progress_display = gr.HTML(
                    update_progress_display(create_progress_steps()),
                    label="📊 İşlem Durumu"
                )
                
                # Log görüntüleyici
                log_display = gr.HTML(
                    get_recent_logs(),
                    label="📜 Sistem Logları"
                )
        
        # Sonuç bölümü
        with gr.Row():
            with gr.Column():
                status_message = gr.Markdown(
                    "Rapor oluşturmak için bir konu girin ve 'Rapor Oluştur' butonuna tıklayın.",
                    label="📋 Durum"
                )
        
        with gr.Row():
            with gr.Column():
                report_output = gr.Markdown(
                    label="📄 Oluşturulan Rapor",
                    elem_classes=["report-section"]
                )
                
                download_output = gr.File(
                    label="💾 Raporu İndir",
                    visible=False,
                    elem_classes=["download-section"]
                )
        
        # Event handlers
        def update_ui_periodically():
            """UI'yi periyodik olarak güncelle"""
            return get_recent_logs()
        
        # Otomatik log güncellemesi
        demo.load(
            update_ui_periodically,
            outputs=log_display,
            every=2  # Her 2 saniyede bir güncelle
        )
        
        # Buton ve enter tuşu olayları
        generate_button.click(
            run_report,
            inputs=topic_input,
            outputs=[status_message, report_output, download_output, progress_display, log_display]
        )
        
        topic_input.submit(
            run_report,
            inputs=topic_input,
            outputs=[status_message, report_output, download_output, progress_display, log_display]
        )
        
        # Dosya indirme durumunu güncelle
        def update_download_visibility(file_path):
            if file_path:
                return gr.update(visible=True, value=file_path)
            return gr.update(visible=False)
        
        download_output.change(
            update_download_visibility,
            inputs=download_output,
            outputs=download_output
        )

    return demo

def launch():
    """Gelişmiş arayüzü başlat"""
    
    # Logging setup
    setup_logging()
    
    print("🚀 NVIDIA Rapor Ajanı başlatılıyor...")
    print("📋 Özellikler:")
    print("   - Real-time progress tracking")
    print("   - Modern UI tasarımı") 
    print("   - Sistem log görüntüleme")
    print("   - Gelişmiş hata yönetimi")
    print("   - Otomatik dosya kaydetme")
    
    demo = build_interface()
    
    # Queue ayarları
    queue_kwargs = {}
    queue_params = inspect.signature(gr.Blocks.queue).parameters
    
    if "default_concurrency_limit" in queue_params:
        queue_kwargs["default_concurrency_limit"] = 1
    elif "concurrency_count" in queue_params:
        queue_kwargs["concurrency_count"] = 1
    
    print("\n🌐 Arayüz açılıyor...")
    demo.queue(**queue_kwargs).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    launch()
