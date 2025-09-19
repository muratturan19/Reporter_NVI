# 🤖 Rapor Ajanı - Kurulum ve Kullanım Kılavuzu

## 📋 Genel Bakış

Bu sistem, modüler sağlayıcı mimarisi ile farklı LLM ve arama servislerini
birleştirerek otomatik rapor oluşturan bir AI ajanıdır. Varsayılan
kombinasyon OpenRouter üzerindeki NVIDIA Nemotron + Tavily aramasıdır; ancak
OpenAI GPT-4o, Anthropic Claude gibi modelleri ve EXA, SerpAPI ya da You.com gibi
arama servislerini de kolayca seçebilirsiniz. Sistem şu bileşenlerden oluşur:

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
# Zorunlu API anahtarları (varsayılan Nemotron + Tavily kombinasyonu için)
OPENROUTER_API_KEY=your_openrouter_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Opsiyonel LLM sağlayıcıları
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_MODEL=claude-3-haiku-20240307

# Opsiyonel arama sağlayıcıları
EXA_API_KEY=your_exa_key_here
SERPAPI_API_KEY=your_serpapi_key_here
YOUCOM_API_KEY=your_youcom_key_here
# Opsiyonel You.com ayarları
# YOUCOM_SAFE_SEARCH=Moderate
# YOUCOM_LANGUAGE=tr
# YOUCOM_COUNTRY=tr
# YOUCOM_DOMAIN=you.com

# Varsayılan sağlayıcı seçimleri (boş bırakırsanız Nemotron + Tavily kullanılır)
DEFAULT_LLM_PROVIDER=openrouter-nemotron
DEFAULT_SEARCH_PROVIDERS=tavily

# Genel ayarlar
MODEL_NAME=nvidia/nemotron-nano-9b-v2:free
MODEL_TEMPERATURE=0.7
MODEL_MAX_TOKENS=2000
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

### Web Arayüzü (Gradio)

Tarayıcı üzerinden konu girip rapor çıktısını görmek için Gradio tabanlı
arayüzü kullanabilirsiniz:

```bash
python ui.py
```

Komut sonrasında konsolda verilen URL'yi ziyaret ederek şu özelliklere
ulaşabilirsiniz:

- LLM ve arama sağlayıcı kombinasyonlarını dropdown menülerinden seçme
- Konu başlığını metin kutusuna girme veya örneklerden seçme
- Oluşan raporu Markdown formatında görüntüleme
- Raporu otomatik kaydedilen `.md` dosyası olarak indirme
- Süreç hakkında durum mesajlarını takip etme
- Sağlayıcıların güçlü yönlerini gösteren tabloyu inceleme

> Not: Varsayılan kombinasyon için `.env` dosyanızda OpenRouter ve Tavily
> anahtarları bulunmalıdır. Diğer LLM veya arama sağlayıcılarını
> kullanabilmek için ilgili opsiyonel API anahtarlarını eklemeyi unutmayın.

### Programmatik Kullanım

```python
import asyncio
from main_report_agent import MainReportAgent

async def create_report():
    agent = MainReportAgent(
        llm_provider_id="openrouter-nemotron",           # opsiyonel: örn. "openai-gpt4"
        search_provider_ids=["tavily", "exa"]             # opsiyonel: birden fazla sağlayıcı
    )
    
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

`create_llm` fonksiyonu, seçtiğiniz sağlayıcıya göre doğru LangChain LLM
nesnesini oluşturur. Sağlayıcı kimliğini belirtmezseniz `.env`
dosyasındaki `DEFAULT_LLM_PROVIDER` değeri kullanılır.

```python
from report_agent_setup import create_llm

# Varsayılan sağlayıcıyı kullan
llm = create_llm()

# Belirli bir sağlayıcı seç
gpt_llm = create_llm("openai-gpt4")
```

### Arama Ayarları

Birden fazla arama servisini aynı anda kullanmak için `create_search_tool`
fonksiyonuna sağlayıcı kimliklerini liste olarak verebilirsiniz. Fonksiyon,
seçili tüm servisleri paralel çalıştıran LangChain uyumlu bir araç döndürür.

```python
from report_agent_setup import create_search_tool

# Varsayılan arama sağlayıcısı (Tavily)
default_search = create_search_tool()

# Tavily + EXA kombinasyonu
multi_search = create_search_tool(["tavily", "exa"])
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