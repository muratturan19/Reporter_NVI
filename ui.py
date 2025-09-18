# -*- coding: utf-8 -*-
"""
Tamamen Yeniden Tasarlanmış Gradio Arayüzü - Modern, Temiz, İşlevsel
"""

from __future__ import annotations

import inspect
import os
import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from queue import Queue

import gradio as gr
from dotenv import load_dotenv

from main_report_agent import MainReportAgent
from provider_manager import ProviderFactory
from report_agent_setup import (
    DEFAULT_LLM_PROVIDER_ID,
    DEFAULT_SEARCH_PROVIDERS,
)

# Ortam değişkenlerini yükle
load_dotenv()

# Global değişkenler
_agent: Optional[MainReportAgent] = None
_agent_config: Optional[Dict[str, Sequence[str]]] = None
_log_queue = Queue()

# Provider seçenekleri
LLM_PROVIDER_OPTIONS = ProviderFactory.get_llm_provider_options()
SEARCH_PROVIDER_OPTIONS = ProviderFactory.get_search_provider_options()
LLM_PROVIDER_MAP: Dict[str, Dict[str, Any]] = {option["id"]: option for option in LLM_PROVIDER_OPTIONS}
SEARCH_PROVIDER_MAP: Dict[str, Dict[str, Any]] = {option["id"]: option for option in SEARCH_PROVIDER_OPTIONS}

def _build_choice_label(option: Dict[str, Any]) -> str:
    status = "Hazır" if option.get("available") else "API Key Gerekli"
    return f"{option['name']} · {status}"

LLM_CHOICES: List[tuple[str, str]] = [
    (_build_choice_label(option), option["id"]) for option in LLM_PROVIDER_OPTIONS
]

SEARCH_CHOICES: List[tuple[str, str]] = [
    (_build_choice_label(option), option["id"]) for option in SEARCH_PROVIDER_OPTIONS
]

# Modern CSS Tasarımı
MODERN_CSS = """
/* Reset ve Base Styles */
* {
    box-sizing: border-box;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Ana Header */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 40px 30px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0 0 10px 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.main-header p {
    font-size: 1.1rem;
    opacity: 0.95;
    margin: 5px 0;
}

/* Ana Grid Layout */
.main-content {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 30px;
    margin-bottom: 30px;
}

/* Input Bölümü */
.input-section {
    background: white;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

.form-group {
    margin-bottom: 25px;
}

.form-group:last-child {
    margin-bottom: 0;
}

/* Provider Cards - Kompakt Tasarım */
.provider-info {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
    height: fit-content;
}

.provider-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 15px 0;
}

.provider-card {
    background: #f8fafc;
    border-radius: 10px;
    padding: 15px;
    border-left: 4px solid #667eea;
    transition: all 0.2s ease;
}

.provider-card:hover {
    background: #f1f5f9;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.provider-name {
    font-size: 14px;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 8px;
    display: block;
}

.provider-status {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
}

.status-ready {
    background: rgba(34, 197, 94, 0.1);
    color: #15803d;
}

.status-needs-key {
    background: rgba(234, 179, 8, 0.1);
    color: #b45309;
}

.provider-features {
    list-style: none;
    padding: 0;
    margin: 8px 0 0 0;
    font-size: 12px;
    color: #64748b;
}

.provider-features li {
    margin-bottom: 4px;
    position: relative;
    padding-left: 12px;
}

.provider-features li::before {
    content: "•";
    color: #667eea;
    font-weight: bold;
    position: absolute;
    left: 0;
}

/* Status Panel */
.status-panel {
    background: white;
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
    margin-bottom: 30px;
}

.status-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

.progress-section, .logs-section {
    background: #f8fafc;
    border-radius: 12px;
    padding: 20px;
}

.progress-section h3, .logs-section h3 {
    margin: 0 0 15px 0;
    color: #1f2937;
    font-size: 16px;
}

/* Progress Steps */
.step {
    display: flex;
    align-items: center;
    padding: 12px;
    margin: 6px 0;
    border-radius: 8px;
    transition: all 0.3s ease;
    font-size: 14px;
}

.step-icon {
    margin-right: 12px;
    width: 20px;
    text-align: center;
}

.step-waiting {
    background: #f1f5f9;
    color: #64748b;
}

.step-active {
    background: #dbeafe;
    color: #1d4ed8;
    border-left: 4px solid #3b82f6;
    transform: translateX(4px);
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

/* Logs Panel */
.logs-container {
    max-height: 200px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    line-height: 1.4;
}

.log-entry {
    padding: 6px 8px;
    margin: 2px 0;
    border-radius: 4px;
    background: white;
    border-left: 3px solid #e5e7eb;
}

.log-info {
    border-left-color: #22c55e;
}

.log-warning {
    border-left-color: #f59e0b;
}

.log-error {
    border-left-color: #ef4444;
}

.log-timestamp {
    color: #64748b;
    margin-right: 8px;
}

.log-level {
    font-weight: 600;
    margin-right: 8px;
}

.log-level.INFO {
    color: #22c55e;
}

.log-level.WARNING {
    color: #f59e0b;
}

.log-level.ERROR {
    color: #ef4444;
}

/* Report Output */
.report-section {
    background: white;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

/* Examples */
.examples-section {
    margin-top: 25px;
}

.example-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 15px;
}

.example-chip {
    background: #f0f9ff;
    color: #0369a1;
    padding: 8px 14px;
    border-radius: 20px;
    border: 1px solid #bae6fd;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.example-chip:hover {
    background: #0369a1;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(3, 105, 161, 0.3);
}

/* Responsive Design */
@media (max-width: 1024px) {
    .main-content {
        grid-template-columns: 1fr;
    }
    
    .status-grid {
        grid-template-columns: 1fr;
    }
    
    .provider-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 768px) {
    .main-header {
        padding: 30px 20px;
    }
    
    .main-header h1 {
        font-size: 2rem;
    }
    
    .input-section, .provider-info, .status-panel, .report-section {
        padding: 20px;
    }
}

/* Animasyonlar */
@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

.animate-slide-up {
    animation: slideInUp 0.6s ease-out;
}

.animate-fade-in {
    animation: fadeIn 0.4s ease-out;
}

/* Custom Scrollbar */
.logs-container::-webkit-scrollbar {
    width: 6px;
}

.logs-container::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 3px;
}

.logs-container::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}

.logs-container::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
"""

class LogCapture(logging.Handler):
    """Gelişmiş log yakalama"""
    
    def emit(self, record):
        log_entry = self.format(record)
        _log_queue.put({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'level': record.levelname,
            'message': log_entry,
            'name': record.name
        })

def setup_logging():
    """Logging kurulumu"""
    handler = LogCapture()
    handler.setFormatter(logging.Formatter('%(message)s'))
    
    loggers = [
        'main_report_agent',
        'researcher_agent', 
        'writer_agent',
        'json_parser_fix',
        'provider_manager'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

def build_provider_cards() -> str:
    """Kompakt provider kartları oluştur"""
    
    # En önemli provider'ları seç
    featured_llm = [p for p in LLM_PROVIDER_OPTIONS if p.get("default")] + [p for p in LLM_PROVIDER_OPTIONS if not p.get("default")]
    featured_search = [p for p in SEARCH_PROVIDER_OPTIONS if p.get("default")] + [p for p in SEARCH_PROVIDER_OPTIONS if not p.get("default")]
    
    featured_llm = featured_llm[:2]  # İlk 2
    featured_search = featured_search[:2]  # İlk 2
    
    def render_card(provider: Dict[str, Any]) -> str:
        status_class = "status-ready" if provider.get("available") else "status-needs-key"
        status_text = "✅ Hazır" if provider.get("available") else "⚠️ API Key"
        
        features = provider.get("strengths", [])[:3]  # İlk 3 özellik
        features_html = ""
        for feature in features:
            features_html += f"<li>{feature}</li>"
        
        return f"""
        <div class="provider-card">
            <span class="provider-name">{provider['name']}</span>
            <span class="provider-status {status_class}">{status_text}</span>
            <ul class="provider-features">{features_html}</ul>
        </div>
        """
    
    llm_cards = "".join([render_card(p) for p in featured_llm])
    search_cards = "".join([render_card(p) for p in featured_search])
    
    return f"""
    <div class="provider-info animate-slide-up">
        <h3 style="margin: 0 0 15px 0; color: #1f2937;">⚙️ Sağlayıcı Durumu</h3>
        
        <div style="margin-bottom: 20px;">
            <h4 style="font-size: 14px; color: #4b5563; margin-bottom: 10px;">🧠 LLM Modelleri</h4>
            <div class="provider-grid">
                {llm_cards}
            </div>
        </div>
        
        <div style="margin-bottom: 15px;">
            <h4 style="font-size: 14px; color: #4b5563; margin-bottom: 10px;">🔍 Arama Araçları</h4>
            <div class="provider-grid">
                {search_cards}
            </div>
        </div>
        
        <div style="background: #fffbeb; border: 1px solid #fed7aa; border-radius: 8px; padding: 12px; font-size: 13px; color: #92400e;">
            💡 <strong>İpucu:</strong> API anahtarları .env dosyasından otomatik okunur
        </div>
    </div>
    """

def create_progress_steps():
    """Progress adımları"""
    return [
        {"id": "init", "text": "Sistem hazırlanıyor", "status": "waiting"},
        {"id": "research", "text": "Web araştırması yapılıyor", "status": "waiting"},
        {"id": "planning", "text": "Rapor yapısı planlanıyor", "status": "waiting"},
        {"id": "writing", "text": "Bölümler yazılıyor", "status": "waiting"},
        {"id": "compiling", "text": "Final rapor derleniyor", "status": "waiting"},
        {"id": "saving", "text": "Rapor kaydediliyor", "status": "waiting"}
    ]

def render_progress_steps(steps):
    """Progress adımlarını HTML olarak render et"""
    icons = {
        'waiting': '⏳',
        'active': '⚡',
        'completed': '✅',
        'error': '❌'
    }
    
    steps_html = ""
    for step in steps:
        icon = icons.get(step['status'], '⏳')
        steps_html += f'''
        <div class="step step-{step['status']}">
            <span class="step-icon">{icon}</span>
            <span>{step["text"]}</span>
        </div>
        '''
    
    return f"""
    <div class="progress-section">
        <h3>📊 İşlem Durumu</h3>
        {steps_html}
    </div>
    """

def render_logs(max_logs=8):
    """Son logları render et"""
    logs = []
    temp_logs = []
    
    # Queue'dan logları al
    while not _log_queue.empty():
        temp_logs.append(_log_queue.get())
    
    recent_logs = temp_logs[-max_logs:] if temp_logs else []
    
    if not recent_logs:
        return """
        <div class="logs-section">
            <h3>📜 Sistem Logları</h3>
            <div class="logs-container">
                <div style="color: #64748b; text-align: center; padding: 20px;">
                    Henüz log yok...
                </div>
            </div>
        </div>
        """
    
    logs_html = ""
    for log in recent_logs:
        level_class = f"log-{log['level'].lower()}"
        logs_html += f'''
        <div class="log-entry {level_class}">
            <span class="log-timestamp">{log["timestamp"]}</span>
            <span class="log-level {log["level"]}">{log["level"]}</span>
            <span>{log["message"]}</span>
        </div>
        '''
    
    return f"""
    <div class="logs-section">
        <h3>📜 Sistem Logları</h3>
        <div class="logs-container">
            {logs_html}
        </div>
    </div>
    """

def _normalize_search_selection(selection: Optional[Sequence[str]]) -> List[str]:
    """Arama sağlayıcı seçimini normalize et"""
    if selection is None:
        return list(DEFAULT_SEARCH_PROVIDERS)
    if isinstance(selection, str):
        normalized = [selection]
    else:
        normalized = [str(item) for item in selection if item]
    return normalized or list(DEFAULT_SEARCH_PROVIDERS)

def _get_agent(llm_provider_id: Optional[str] = None, search_provider_ids: Optional[Sequence[str]] = None) -> MainReportAgent:
    """Agent instance yönetimi"""
    global _agent, _agent_config

    normalized_llm = llm_provider_id or DEFAULT_LLM_PROVIDER_ID
    normalized_search = tuple(_normalize_search_selection(search_provider_ids))
    config_signature = {"llm": normalized_llm, "search": normalized_search}

    if _agent is None or _agent_config != config_signature:
        _agent = MainReportAgent(
            llm_provider_id=normalized_llm,
            search_provider_ids=list(normalized_search),
        )
        _agent_config = config_signature

    return _agent

def _format_provider_display(provider_id: str, provider_map: Dict[str, Dict[str, Any]]) -> str:
    """Provider görüntü formatı"""
    option = provider_map.get(provider_id)
    if not option:
        return provider_id
    status = "✅" if option.get("available") else "⚠️"
    return f"{status} {option['name']}"

def _sanitize_topic(topic: str) -> str:
    """Dosya adı için konu temizleme"""
    safe_chars = [ch if ch.isalnum() else " " for ch in topic.lower()]
    sanitized = "_".join("".join(safe_chars).split())
    if not sanitized:
        sanitized = "rapor"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sanitized[:50]}_{timestamp}.md"

async def run_report(topic: str, llm_provider_id: Optional[str], search_provider_ids: Optional[Sequence[str]]):
    """Ana rapor oluşturma fonksiyonu"""
    
    cleaned_topic = (topic or "").strip()
    selected_llm = llm_provider_id or DEFAULT_LLM_PROVIDER_ID
    selected_search = _normalize_search_selection(search_provider_ids)

    provider_summary = (
        f"LLM: {_format_provider_display(selected_llm, LLM_PROVIDER_MAP)}\n"
        f"Arama: {', '.join([_format_provider_display(pid, SEARCH_PROVIDER_MAP) for pid in selected_search])}"
    )

    if not cleaned_topic:
        steps = create_progress_steps()
        yield (
            f"❌ Lütfen bir rapor konusu girin.\n\n{provider_summary}",
            "",
            None,
            render_progress_steps(steps),
            render_logs()
        )
        return

    # Progress başlat
    steps = create_progress_steps()
    
    # Adım 1: Başlatma
    steps[0]["status"] = "active"
    yield (
        f"🚀 Sistem başlatılıyor...\n\n{provider_summary}",
        "",
        None,
        render_progress_steps(steps),
        render_logs()
    )

    try:
        agent = _get_agent(selected_llm, selected_search)
        steps[0]["status"] = "completed"

        # Adım 2: Araştırma
        steps[1]["status"] = "active"
        yield (
            "🔍 Web araştırması başladı...",
            "",
            None,
            render_progress_steps(steps),
            render_logs()
        )

        # Rapor oluşturma
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
                render_progress_steps(steps),
                render_logs()
            )
            return

        # Tüm adımları tamamlandı olarak işaretle
        for step in steps[:-1]:
            step["status"] = "completed"

        steps[-1]["status"] = "active"
        yield (
            "💾 Rapor kaydediliyor...",
            report,
            None,
            render_progress_steps(steps),
            render_logs()
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
            render_progress_steps(steps),
            render_logs()
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
            render_progress_steps(steps),
            render_logs()
        )

def build_interface() -> gr.Blocks:
    """Ana arayüz oluşturma"""
    
    with gr.Blocks(
        title="NVIDIA Rapor Ajanı", 
        css=MODERN_CSS, 
        theme=gr.themes.Soft(),
        head='<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    ) as demo:

        # Ana Header
        gr.HTML("""
        <div class="main-header animate-fade-in">
            <h1>🧠 NVIDIA Rapor Ajanı</h1>
            <p>Yapay zeka destekli araştırma ve rapor oluşturma sistemi</p>
            <p><em>Web araştırması yaparak kapsamlı Markdown raporları oluşturur</em></p>
        </div>
        """)

        # Ana İçerik Grid
        with gr.Row(elem_classes=["main-content"]):
            with gr.Column(scale=2, elem_classes=["input-section", "animate-slide-up"]):
                # Form elemanları
                llm_dropdown = gr.Dropdown(
                    choices=LLM_CHOICES,
                    value=DEFAULT_LLM_PROVIDER_ID,
                    label="🧠 LLM Sağlayıcısı",
                    info="Raporun yazımında kullanılacak büyük dil modelini seçin",
                    elem_classes=["form-group"]
                )

                search_dropdown = gr.Dropdown(
                    choices=SEARCH_CHOICES,
                    value=list(DEFAULT_SEARCH_PROVIDERS),
                    label="🔎 Arama Sağlayıcıları",
                    info="Bir veya birden fazla web arama sağlayıcısı seçin",
                    multiselect=True,
                    elem_classes=["form-group"]
                )

                topic_input = gr.Textbox(
                    label="📝 Rapor Konusu",
                    placeholder="Örn. 'Yapay zeka ajanlarının sağlık sektöründeki uygulamaları'",
                    lines=3,
                    max_lines=5,
                    elem_classes=["form-group"]
                )
                
                generate_button = gr.Button(
                    "🚀 Rapor Oluştur", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["form-group"]
                )
                
                # Örnek konular
                with gr.Column(elem_classes=["examples-section"]):
                    gr.Markdown("### 💡 Örnek Konular")
                    gr.Examples(
                        examples=[
                            ["Yapay zeka destekli müşteri hizmetleri çözümleri"],
                            ["Sürdürülebilir enerji yönetiminde dijital ikiz uygulamaları"], 
                            ["Finans sektöründe büyük dil modellerinin kullanımı"],
                            ["Endüstri 4.0 ve IoT sensörlerin üretim optimizasyonu"],
                            ["Blockchain teknolojisinin tedarik zinciri yönetimindeki rolü"]
                        ],
                        inputs=[topic_input],
                        elem_classes=["example-chips"]
                    )
            
            with gr.Column(scale=1):
                # Provider bilgileri
                gr.HTML(build_provider_cards())

        # Status Panel
        with gr.Column(elem_classes=["status-panel"]):
            with gr.Row(elem_classes=["status-grid"]):
                progress_display = gr.HTML(
                    render_progress_steps(create_progress_steps()),
                    elem_id="progress-display"
                )
                
                log_display = gr.HTML(
                    render_logs(),
                    elem_id="log-display"
                )
        
        # Sonuç bölümü
        with gr.Column():
            status_message = gr.Markdown(
                "Rapor oluşturmak için yukarıdan bir konu seçin veya girin, sonra **Rapor Oluştur** butonuna tıklayın.",
                elem_classes=["animate-fade-in"]
            )
        
        with gr.Column(elem_classes=["report-section"]):
            report_output = gr.Markdown(
                label="📄 Oluşturulan Rapor"
            )
            
            download_output = gr.File(
                label="💾 Raporu İndir",
                visible=False
            )

        # Event handlers
        def update_ui_periodically():
            return render_logs()
        
        # Auto refresh
        demo.load(update_ui_periodically, outputs=[log_display])
        
        # Timer for log updates
        log_timer = gr.Timer(value=3)
        log_timer.tick(update_ui_periodically, outputs=[log_display])
        
        # Main button events
        generate_button.click(
            run_report,
            inputs=[topic_input, llm_dropdown, search_dropdown],
            outputs=[status_message, report_output, download_output, progress_display, log_display]
        )

        topic_input.submit(
            run_report,
            inputs=[topic_input, llm_dropdown, search_dropdown],
            outputs=[status_message, report_output, download_output, progress_display, log_display]
        )
        
        # Download visibility handler
        def update_download_visibility(file_path):
            if file_path:
                return gr.update(visible=True, value=file_path)
            return gr.update(visible=False)
        
        download_output.change(
            update_download_visibility,
            inputs=[download_output],
            outputs=[download_output]
        )

    return demo

def launch():
    """Gelişmiş arayüz başlatma"""
    
    # Logging kurulum
    setup_logging()
    
    print("🚀 NVIDIA Rapor Ajanı başlatılıyor...")
    print("✨ Özellikler:")
    print("   - Modern ve responsive UI tasarımı")
    print("   - Real-time progress tracking")
    print("   - Gelişmiş sistem log görüntüleme") 
    print("   - Multi-provider support")
    print("   - Otomatik dosya kaydetme")
    print("   - Kompakt sağlayıcı bilgileri")
    
    demo = build_interface()
    
    # Queue ayarları
    queue_kwargs = {}
    queue_params = inspect.signature(gr.Blocks.queue).parameters
    
    if "default_concurrency_limit" in queue_params:
        queue_kwargs["default_concurrency_limit"] = 1
    elif "concurrency_count" in queue_params:
        queue_kwargs["concurrency_count"] = 1
    
    print("\n🌐 Arayüz açılıyor...")
    print("📍 Adres: http://127.0.0.1:7860")
    
    demo.queue(**queue_kwargs).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    launch()
