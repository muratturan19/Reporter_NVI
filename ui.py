from __future__ import annotations

import gradio as gr
from textwrap import dedent

HTML_CONTENT = dedent(
    """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 40px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* Main Content Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        /* Input Section */
        .input-section {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #2d3748;
        }

        .dropdown-select, .text-input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: white;
        }

        .dropdown-select:focus, .text-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }

        .text-input {
            resize: vertical;
            min-height: 80px;
        }

        .generate-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        }

        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }

        /* Provider Info - Compact */
        .provider-info-compact {
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            height: fit-content;
        }

        .provider-toggle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            padding: 10px;
            border-radius: 8px;
            transition: background 0.2s ease;
        }

        .provider-toggle:hover {
            background: #f8fafc;
        }

        .provider-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .provider-content.expanded {
            max-height: 800px;
        }

        .provider-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }

        .provider-card {
            background: #f8fafc;
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #667eea;
        }

        .provider-name {
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 5px;
        }

        .provider-status {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            margin-bottom: 8px;
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
            font-size: 13px;
            color: #64748b;
        }

        .provider-features li {
            margin-bottom: 3px;
        }

        .provider-features li::before {
            content: "• ";
            color: #667eea;
            font-weight: bold;
        }

        /* Status Panel */
        .status-panel {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .progress-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .progress-steps, .logs-panel {
            background: #f8fafc;
            border-radius: 12px;
            padding: 20px;
        }

        .step {
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

        .step-icon {
            margin-right: 12px;
            font-size: 16px;
        }

        /* Examples Section */
        .examples {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-top: 25px;
        }

        .examples h4 {
            margin-bottom: 15px;
            color: #2d3748;
        }

        .example-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .example-chip {
            background: #f0f9ff;
            color: #0369a1;
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid #bae6fd;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        .example-chip:hover {
            background: #0369a1;
            color: white;
        }

        /* Report Output */
        .report-output {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            margin-top: 30px;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }

            .progress-container {
                grid-template-columns: 1fr;
            }

            .provider-grid {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 2rem;
            }
        }

        /* Animations */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .animate-in {
            animation: slideIn 0.6s ease-out;
        }
    </style>
    <div class="container">
        <div class="header animate-in">
            <h1>🧠 NVIDIA Rapor Ajanı</h1>
            <p>Yapay zeka destekli araştırma ve rapor oluşturma sistemi</p>
        </div>
        <div class="main-grid">
            <div class="input-section animate-in">
                <div class="form-group">
                    <label>🧠 LLM Sağlayıcısı</label>
                    <select class="dropdown-select">
                        <option>OpenRouter · NVIDIA Nemotron · Hazır</option>
                        <option>OpenAI · GPT-4o · API anahtarı gerekli</option>
                        <option>Anthropic · Claude 3 · API anahtarı gerekli</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>🔎 Arama Sağlayıcıları</label>
                    <select class="dropdown-select" multiple>
                        <option selected>Tavily Search · Hazır</option>
                        <option>Exa Semantic Search · API anahtarı gerekli</option>
                        <option>SerpAPI Google Search · API anahtarı gerekli</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>📝 Rapor Konusu</label>
                    <textarea class="text-input" placeholder="Örn. 'Yapay zeka ajanlarının sağlık sektöründeki uygulamaları'"></textarea>
                </div>
                <button class="generate-btn">🚀 Rapor Oluştur</button>
                <div class="examples">
                    <h4>💡 Örnek Konular</h4>
                    <div class="example-chips">
                        <span class="example-chip">AI müşteri hizmetleri</span>
                        <span class="example-chip">Dijital ikiz uygulamaları</span>
                        <span class="example-chip">Finans sektöründe LLM</span>
                        <span class="example-chip">Endüstri 4.0 IoT</span>
                        <span class="example-chip">Blockchain tedarik zinciri</span>
                    </div>
                </div>
            </div>
            <div class="provider-info-compact animate-in">
                <div class="provider-toggle" onclick="toggleProviders()">
                    <h3>⚙️ Sağlayıcı Detayları</h3>
                    <span id="toggle-icon">▼</span>
                </div>
                <div class="provider-content" id="provider-content">
                    <div class="provider-grid">
                        <div class="provider-card">
                            <div class="provider-name">NVIDIA Nemotron</div>
                            <span class="provider-status status-ready">✅ Hazır</span>
                            <ul class="provider-features">
                                <li>Open-source model</li>
                                <li>Türkçe desteği</li>
                                <li>Uygun maliyet</li>
                            </ul>
                        </div>
                        <div class="provider-card">
                            <div class="provider-name">Tavily Search</div>
                            <span class="provider-status status-ready">✅ Hazır</span>
                            <ul class="provider-features">
                                <li>Otomatik özet</li>
                                <li>Hızlı yanıt</li>
                                <li>AI-optimize</li>
                            </ul>
                        </div>
                        <div class="provider-card">
                            <div class="provider-name">OpenAI GPT-4</div>
                            <span class="provider-status status-needs-key">⚠️ API Key</span>
                            <ul class="provider-features">
                                <li>Üstün muhakeme</li>
                                <li>Geniş entegrasyon</li>
                                <li>Çok dilli</li>
                            </ul>
                        </div>
                        <div class="provider-card">
                            <div class="provider-name">EXA Semantic</div>
                            <span class="provider-status status-needs-key">⚠️ API Key</span>
                            <ul class="provider-features">
                                <li>Semantik arama</li>
                                <li>Kaynak çeşitliliği</li>
                                <li>Autoprompt</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="status-panel">
            <div class="progress-container">
                <div class="progress-steps">
                    <h4>📊 İşlem Durumu</h4>
                    <div class="step step-waiting">
                        <span class="step-icon">⏳</span>
                        <span>Sistem hazırlanıyor</span>
                    </div>
                    <div class="step step-waiting">
                        <span class="step-icon">⏳</span>
                        <span>Web araştırması yapılıyor</span>
                    </div>
                    <div class="step step-waiting">
                        <span class="step-icon">⏳</span>
                        <span>Rapor yapısı planlanıyor</span>
                    </div>
                    <div class="step step-waiting">
                        <span class="step-icon">⏳</span>
                        <span>Bölümler yazılıyor</span>
                    </div>
                    <div class="step step-waiting">
                        <span class="step-icon">⏳</span>
                        <span>Final rapor derleniyor</span>
                    </div>
                </div>
                <div class="logs-panel">
                    <h4>📜 Sistem Logları</h4>
                    <div style="color: #64748b; font-size: 14px; font-family: monospace;">
                        Henüz log yok...
                    </div>
                </div>
            </div>
        </div>
        <div class="report-output">
            <h3>📄 Oluşturulan Rapor</h3>
            <p style="color: #64748b; margin-top: 10px;">
                Rapor oluşturmak için bir konu girin ve 'Rapor Oluştur' butonuna tıklayın.
            </p>
        </div>
    </div>
    <script>
        function toggleProviders() {
            const content = document.getElementById('provider-content');
            const icon = document.getElementById('toggle-icon');

            if (content.classList.contains('expanded')) {
                content.classList.remove('expanded');
                icon.textContent = '▼';
            } else {
                content.classList.add('expanded');
                icon.textContent = '▲';
            }
        }

        document.querySelectorAll('.example-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                document.querySelector('.text-input').value = chip.textContent;
            });
        });

        function simulateProgress() {
            const steps = document.querySelectorAll('.step');
            let current = 0;

            const interval = setInterval(() => {
                if (current > 0) {
                    steps[current - 1].className = 'step step-completed';
                    steps[current - 1].querySelector('.step-icon').textContent = '✅';
                }

                if (current < steps.length) {
                    steps[current].className = 'step step-active';
                    steps[current].querySelector('.step-icon').textContent = '⚡';
                    current++;
                } else {
                    clearInterval(interval);
                }
            }, 1000);
        }

        document.querySelector('.generate-btn').addEventListener('click', simulateProgress);
    </script>
    """
)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="NVIDIA Rapor Ajanı - İyileştirilmiş UI") as demo:
        gr.HTML(HTML_CONTENT)
    return demo


def launch():
    demo = build_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    launch()
