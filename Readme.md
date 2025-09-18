# 🤖 Rapor Ajanı - Kurulum ve Kullanım Kılavuzu

## 📋 Genel Bakış

Bu sistem, NVIDIA Nemotron modeli kullanarak otomatik rapor oluşturan bir AI ajanıdır. Sistem şu bileşenlerden oluşur:

- **Araştırmacı Ajan**: Web araştırması yapar
- **Yazar Ajan**: Araştırma verilerini kullanarak bölümler yazar  
- **Ana Ajan**: Tüm süreci yönetir ve final raporu derler

## 🛠️ Kurulum Adımları

### 1. Sistem Gereksinimleri
- **İşletim Sistemi**: Windows 10/11
- **Python**: 3.8 veya üzeri
- **Encoding**: UTF-8 desteği

### 2. Python Kütüphanelerini Yükleyin

```bash
pip install langchain-nvidia-ai-endpoints
pip install langgraph
pip install langchain-core
pip install httpx
pip install typing-extensions
pip install asyncio
```

### 3. API Keys Alın ve .env Dosyası Oluşturun

#### OpenRouter API Key
1. [OpenRouter](https://openrouter.ai/) sitesine gidin
2. Hesap oluşturun
3. API key alın
4. Credits satın alın (NVIDIA Nemotron için)

#### Tavily API Key
1. [Tavily](https://app.tavily.com/) sitesine gidin
2. Hesap oluşturun
3. API key alın

#### .env Dosyası Oluşturun
Proje klasöründe `.env` adında bir dosya oluşturun:

```env
# API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Model ayarları (opsiyonel)
MODEL_NAME=nvidia/nemotron-nano-9b-v2:free
MODEL_TEMPERATURE=0.7
MODEL_MAX_TOKENS=2000

# Arama ayarları (opsiyonel)
SEARCH_MAX_RESULTS=5
DEFAULT_SEARCH_QUERIES=3
```

**Güvenlik Notu**: `.env` dosyasını `.gitignore` dosyasına ekleyin!

### 4. Dosya Yapısını Oluşturun

```
rapor_ajanı/
├── .env                       # API keys ve ayarlar
├── .gitignore                 # .env dosyasını gizle
├── report_agent_setup.py      # Temel kurulum
├── researcher_agent.py        # Araştırmacı ajan
├── writer_agent.py            # Yazar ajan  
├── main_report_agent.py       # Ana sistem
├── requirements.txt           # Kütüphane listesi
└── raporlar/                  # Oluşturulan raporlar (otomatik)
```

#### .gitignore Dosyası Oluşturun
Güvenlik için `.gitignore` dosyası oluşturun:

```gitignore
# Environment variables
.env

# Log dosyaları
*.log

# Python cache
__pycache__/
*.pyc
*.pyo

# Raporlar (opsiyonel)
raporlar/
```

## 🚀 Kullanım

### İnteraktif Mod

```bash
python main_report_agent.py
```

Program başladığında:
1. Rapor oluşturmak istediğiniz konuyu girin
2. Sistem otomatik olarak:
   - Web araştırması yapar
   - Rapor yapısını planlar
   - Bölümleri yazar
   - Final raporu derler
3. Raporu kaydetme seçeneği sunar

### Programmatik Kullanım

```python
import asyncio
from main_report_agent import MainReportAgent

async def create_report():
    agent = MainReportAgent()
    
    # Rapor oluştur
    report = await agent.generate_report(
        "Yapay zeka ajanlarının sağlık sektöründeki uygulamaları"
    )
    
    # Raporu kaydet
    filename = await agent.save_report(report)
    print(f"Rapor kaydedildi: {filename}")

# Çalıştır
asyncio.run(create_report())
```

## 🔧 Yapılandırma

### Model Ayarları

`report_agent_setup.py` dosyasında:

```python
llm = ChatNVIDIA(
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-nano-9b-v2:free",  # Model seçimi
    api_key=OPENROUTER_API_KEY,
    temperature=0.7,    # Yaratıcılık seviyesi (0-1)
    max_tokens=2000     # Maksimum çıktı uzunluğu
)
```

### Arama Ayarları

`researcher_agent.py` dosyasında:

```python
# Araştırma sorgu sayısı
number_of_queries = 3  # Varsayılan: 3

# Arama sonuç sayısı
max_results = 5        # Varsayılan: 5
```

## 📊 Örnek Çıktı

```markdown
# Yapay Zeka Ajanlarının Sağlık Sektöründeki Uygulamaları

*Oluşturulma Tarihi: 2025-01-09*

## İçindekiler
1. Giriş
2. Temel Kavramlar
3. Sağlık Sektöründeki Uygulamalar
4. Avantajlar ve Zorluklar
5. Gelecek Perspektifleri
6. Sonuç ve Öneriler

## 1. Giriş

Yapay zeka ajanları, sağlık sektöründe devrim yaratmaktadır...

[Rapor devam eder...]
```

## 🐛 Sorun Giderme

### Yaygın Hatalar

#### 1. .env Dosyası Bulunamadı
```
❌ Eksik API key'ler: OPENROUTER_API_KEY, TAVILY_API_KEY
```
**Çözüm**: `.env` dosyasının proje kök dizininde olduğundan emin olun.

#### 2. API Key Hatası
```
Model yükleme hatası: Invalid API key
```
**Çözüm**: API key'lerinizin doğru ve aktif olduğundan emin olun.

#### 3. .env Dosyası Okunmuyor
**Çözüm**: `python-dotenv` kütüphanesinin yüklü olduğundan emin olun:
```bash
pip install python-dotenv
```

### Log Dosyaları

Sistem `report_agent.log` dosyasına detaylı loglar yazar. Hata durumunda bu dosyayı kontrol edin.

## 🎯 İpuçları

### Daha İyi Sonuçlar İçin

1. **Spesifik konular seçin**: "AI" yerine "Sağlık sektöründe AI ajanları"
2. **Türkçe anahtar kelimeler kullanın**: Sistem hem Türkçe hem İngilizce kaynaklarda arama yapar
3. **Sabırlı olun**: Kapsamlı raporlar 3-5 dakika sürebilir
4. **API limitlerini kontrol edin**: OpenRouter ve Tavily'nin rate limitleri vardır

### Özelleştirme

- **Prompt'ları düzenleyin**: Yazım stilini değiştirmek için prompt'ları güncelleyin
- **Bölüm sayısını ayarlayın**: `REPORT_PLANNER_PROMPT` içinde bölüm sayısını değiştirin
- **Arama konularını genişletin**: `search_web` fonksiyonuna yeni kategoriler ekleyin

## 📞 Destek

Sorun yaşarsanız:
1. Log dosyalarını kontrol edin
2. API key'lerinizi doğrulayın  
3. İnternet bağlantınızı test edin
4. Kütüphane versiyonlarını güncelleyin

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir. Ticari kullanım için API sağlayıcılarının koşullarını kontrol edin.

---

**Not**: Bu sistem Windows sistemleri için optimize edilmiştir ve UTF-8 encoding destekler. Türkçe karakterler sorunsuz çalışır.